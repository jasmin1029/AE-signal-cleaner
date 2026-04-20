#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花岗岩单轴压缩试验 - 综合分析 v6
核心改进：区分 "稳态干扰" 与 "破坏期真实AE"
  - 稳态干扰：60 dB hits 以 US 周期速率稳定出现（低局部速率）→ 全段振幅带删除
  - 破坏期真实AE：60 dB hits 夹在爆发性高速率撞击中 → 仅用时间窗，保留真实AE
  判断流程：
    Step-1  基础时间窗（50ms/500ms）锁定确认干扰 + 拟合干扰振幅带
    Step-2  滑窗计算局部撞击速率；识别"爆发段"
    Step-3  非爆发段：振幅带全段删除（干扰主导）
            爆发段：仅时间窗（真实AE主导，振幅带删除暂停）
    最终干扰 = time_window | (in_amp_band AND NOT in_burst)

Sample: Granite Ph50mm x H100mm  |  Date: 2026-04-15
"""

import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
import warnings, os, gc

warnings.filterwarnings('ignore')

# ─── 中文字体 ─────────────────────────────────────────────────────────────
for _f in ['Microsoft YaHei','SimHei','WenQuanYi Micro Hei','Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams['font.family'] = _f; break
    except: pass
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
    'legend.fontsize':9,'axes.grid':True,'grid.alpha':0.25,'grid.linewidth':0.5,
    'axes.spines.top':False,'axes.spines.right':False,
    'figure.dpi':150,'savefig.dpi':200,'savefig.bbox':'tight'})

# ─── 路径 ─────────────────────────────────────────────────────────────────
BASE     = r'g:\Cursor project\ZCY-shengfashe'
US_FILE  = os.path.join(BASE,'超声波','04-15 - ultrasonics data.csv')
CAL_FILE = os.path.join(BASE,'超声波','chushi.csv')
AE_HITS  = os.path.join(BASE,'声发射','04-15-hits-振铃计数、能量等.TXT')
AE_EVTS  = os.path.join(BASE,'声发射','04-15-声发射事件.TXT')
RESULT_DIR = os.path.join(BASE,'结果')
os.makedirs(RESULT_DIR, exist_ok=True)

# ─── 参数 ─────────────────────────────────────────────────────────────────
H_MM = 100.0;  H_M = H_MM/1000.0
FS_HZ = 40e6
BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER = 50e3, 700e3, 4
AIC_SEARCH_OFFSET_US = 12.0
AIC_SEARCH_WIDTH_US  = 30.0
AIC_GLOBAL_START_US  = 5.0

US_MASK_PRE  = 0.05    # 基础时间窗：激发前 50 ms
US_MASK_POST = 0.50    # 基础时间窗：激发后 500 ms

T_EARLY    = 200.0    # 用前 200s 拟合干扰振幅带
BND_FRAC   = 0.30     # 振幅带截断高度（峰高的 30% ≈ FWHM）

RATE_BIN   = 5.0      # 速率统计分箱宽度（s）
BURST_FACTOR = 5.0    # 局部速率 > 早期基准速率 × BURST_FACTOR → 爆发段

COLORS    = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
CH_COLORS = {i+1:COLORS[i] for i in range(6)}


# ═══════════════════════════════════════════════════════════════════════════
# § 0  工具函数
# ═══════════════════════════════════════════════════════════════════════════
def bp_sos(fs,lo,hi,order=4):
    nyq=fs/2; return butter(order,[lo/nyq,hi/nyq],btype='band',output='sos')
def apply_bp(s,sos):
    try: return sosfiltfilt(sos,s.astype(float))
    except: return s.astype(float)
def aic_pick(wf,t,s0,s1):
    N=len(wf); i0=max(1,int(np.searchsorted(t,s0))); i1=min(N-1,int(np.searchsorted(t,s1)))
    if i1<=i0+2: return np.nan,None,(i0,i1)
    x=wf-wf.mean(); cs=np.cumsum(x*x); tot=cs[-1]+1e-30
    k=np.arange(i0,i1)
    v1=np.maximum(cs[k-1]/k,1e-30); v2=np.maximum((tot-cs[k-1])/(N-k),1e-30)
    aic_s=np.convolve(k*np.log(v1)+(N-k)*np.log(v2),np.ones(5)/5,mode='same')
    return t[k[np.argmin(aic_s)]],aic_s,(i0,i1)


# ═══════════════════════════════════════════════════════════════════════════
# § 1  超声波 AIC 拾取 + Vp
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("加载超声波数据...")
us_raw   = pd.read_csv(US_FILE,header=None,low_memory=False,dtype=str)
us_ts    = pd.to_numeric(us_raw.iloc[2,1:],errors='coerce').dropna().values
n_sw     = len(us_ts)
us_pt_sw = pd.to_numeric(us_raw.iloc[5,1:n_sw+1],errors='coerce').values
wf_block   = us_raw.iloc[7:,:]
wf_time_us = pd.to_numeric(wf_block.iloc[:,0],errors='coerce').values
wf_data    = wf_block.iloc[:,1:n_sw+1].apply(pd.to_numeric,errors='coerce').values
del us_raw,wf_block; gc.collect()
n_samp = wf_data.shape[0]
print(f"  {n_sw} sweeps  {n_samp} pts  {us_ts[0]:.1f}-{us_ts[-1]:.1f} s")

sos_bp = bp_sos(FS_HZ,BP_LOW_HZ,BP_HIGH_HZ,BP_ORDER)

# 对零校准
print("对零校准...")
t_cal_aic = np.nan
try:
    cal = pd.read_csv(CAL_FILE,header=None,skiprows=154,low_memory=False,dtype=str,encoding='latin-1')
    ct = pd.to_numeric(cal.iloc[:,0],errors='coerce').values
    cs = pd.to_numeric(cal.iloc[:,1],errors='coerce').values
    ok = ~(np.isnan(ct)|np.isnan(cs)); ct,cs = ct[ok],cs[ok]
    snr = np.max(np.abs(cs))/(np.std(cs[:100])+1e-30)
    if snr>10:
        t_cal_aic,_,_ = aic_pick(apply_bp(cs,sos_bp),ct,0.5,40.0)
    print(f"  SNR={snr:.0f}  t_cal={t_cal_aic:.3f} us")
except Exception as e:
    print(f"  警告: {e}")
if np.isnan(t_cal_aic):
    es = us_pt_sw[~np.isnan(us_pt_sw)][:20]
    t_cal_aic = float(np.nanmedian(es)-H_M/4800.0*1e6) if len(es) else 0.0
sys_delay = t_cal_aic

print("AIC 拾取...")
t_arr = wf_time_us.copy()
us_pt_aic = np.full(n_sw,np.nan)
for i in range(n_sw):
    wf=wf_data[:,i].astype(float)
    if np.sum(np.isnan(wf))>n_samp*0.5: continue
    wf[np.isnan(wf)]=0.0
    ref=us_pt_sw[i] if not np.isnan(us_pt_sw[i]) else 30.0
    us_pt_aic[i],_,_=aic_pick(apply_bp(wf,sos_bp),t_arr,
                               max(AIC_GLOBAL_START_US,ref-AIC_SEARCH_OFFSET_US),
                               ref+AIC_SEARCH_WIDTH_US)

valid_aic  = ~np.isnan(us_pt_aic)
travel_aic = (us_pt_aic-sys_delay)*1e-6
us_Vp_aic  = np.where((travel_aic>5e-6)&(travel_aic<200e-6),H_M/travel_aic/1000.,np.nan)
travel_sw  = (us_pt_sw-sys_delay)*1e-6
us_Vp_sw   = np.where((travel_sw>5e-6)&(travel_sw<200e-6),H_M/travel_sw/1000.,np.nan)
diff_valid = (us_pt_aic-us_pt_sw)[valid_aic&~np.isnan(us_pt_sw)]
print(f"  Vp(AIC): {np.nanmin(us_Vp_aic):.2f}-{np.nanmax(us_Vp_aic):.2f} km/s")

vp_base  = np.nanmedian(us_Vp_aic[valid_aic][:30]) if valid_aic.sum()>=30 else np.nan
fail_mask= (~np.isnan(us_Vp_aic))&(us_Vp_aic<vp_base*0.7) if not np.isnan(vp_base) else np.zeros(n_sw,bool)
t_fail   = float(us_ts[np.where(fail_mask)[0][0]]) if np.any(fail_mask) else np.nan
print(f"  推测破坏时刻: {t_fail:.1f} s" if not np.isnan(t_fail) else "  未检测到明显破坏时刻")


# ═══════════════════════════════════════════════════════════════════════════
# § 2  声发射数据
# ═══════════════════════════════════════════════════════════════════════════
print("\n加载声发射数据...")
def parse_ae_hits(path):
    rows=[]
    with open(path,'r',errors='replace') as fh:
        for line in fh:
            p=line.strip().split()
            if len(p)>=9:
                try: rows.append({'Time':float(p[1]),'CH':int(p[2]),'RISE':int(p[3]),
                    'COUN':int(p[4]),'ENER':int(p[5]),'DURATION':int(p[6]),
                    'AMP':float(p[7]),'ABS_E':float(p[8])})
                except: pass
    return pd.DataFrame(rows)

ae    = parse_ae_hits(AE_HITS)
ae    = ae[ae['Time']>0].sort_values('Time').reset_index(drop=True)
ae_t  = ae['Time'].values; ae_amp = ae['AMP'].values; ae_ch = ae['CH'].values
n_hits= len(ae)
print(f"  {n_hits} hits  {ae_t.min():.1f}-{ae_t.max():.1f} s")

def parse_ae_events(path):
    events,cur=[],None
    with open(path,'r',errors='replace') as fh:
        for line in fh:
            line=line.strip()
            if line.startswith('* Gp#'):
                try:
                    x=float(line.split('x,y,z =')[1].split(',')[0])
                    y=float(line.split('x,y,z =')[1].split(',')[1])
                    z=float(line.split('x,y,z =')[1].split(',')[2].split(',')[0])
                    cur={'x':x,'y':y,'z':z,'time':None}; events.append(cur)
                except: cur=None
            elif line.startswith('*') and cur is not None:
                p=line.lstrip('*').split()
                if len(p)>=8:
                    try:
                        t=float(p[0])
                        if cur['time'] is None: cur['time']=t
                    except: pass
    return pd.DataFrame([e for e in events if e['time'] is not None])

evts = parse_ae_events(AE_EVTS)
print(f"  {len(evts)} 已定位事件")


# ═══════════════════════════════════════════════════════════════════════════
# § 3  干扰识别（v6）
# ═══════════════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("干扰识别 v6（爆发段保护 + 振幅带全段删除）...")

period_approx = float(np.median(np.diff(us_ts)))
us_ts_sorted  = np.sort(us_ts)
_idx  = np.searchsorted(us_ts_sorted,ae_t,side='right')-1
dt_from_us = np.where(_idx>=0, ae_t-us_ts_sorted[np.maximum(_idx,0)], np.nan)
_idx_n     = np.searchsorted(us_ts_sorted,ae_t,side='left')
dt_to_next = np.where(_idx_n<len(us_ts),
                       us_ts_sorted[np.minimum(_idx_n,len(us_ts)-1)]-ae_t, np.inf)

# ── Step-1  基础时间窗 + 振幅带拟合 ──────────────────────────────────────
contam_tw = ((dt_from_us>=-US_MASK_PRE)&(dt_from_us<=US_MASK_POST)) | \
            ((dt_to_next>=0)&(dt_to_next<=US_MASK_PRE))

# 用前 T_EARLY 秒内时间窗撞击拟合干扰振幅带
amp_fit = ae_amp[contam_tw & (ae_t < T_EARLY)]
if len(amp_fit) < 30:
    amp_fit = ae_amp[contam_tw]
    print(f"  振幅带拟合: 使用全段时间窗 (n={len(amp_fit)})")
else:
    print(f"  振幅带拟合: 使用前 {T_EARLY:.0f}s 时间窗 (n={len(amp_fit)})")

ae_fine = np.arange(35,106,1); ac_fine = (ae_fine[:-1]+ae_fine[1:])/2
hist_fit,_ = np.histogram(amp_fit, bins=ae_fine)
hs = gaussian_filter1d(hist_fit.astype(float), sigma=2)
pk = int(np.argmax(hs)); amp_peak = float(ac_fine[pk]); pv = hs[pk]

li = pk
while li>0 and hs[li]>pv*BND_FRAC: li-=1
ri = pk
while ri<len(hs)-1 and hs[ri]>pv*BND_FRAC: ri+=1
AMP_BAND_LOW  = float(ac_fine[max(li-1,0)])
AMP_BAND_HIGH = float(ac_fine[min(ri+1,len(ac_fine)-1)])
print(f"  干扰振幅带: [{AMP_BAND_LOW:.1f}, {AMP_BAND_HIGH:.1f}] dB (峰值 {amp_peak:.1f} dB)")

in_amp_band = (ae_amp>=AMP_BAND_LOW)&(ae_amp<=AMP_BAND_HIGH)

# ── Step-2  滑窗局部撞击速率 → 识别爆发段 ─────────────────────────────
# 以 RATE_BIN 秒为分箱单元统计全通道撞击速率
t_bins_r  = np.arange(0, ae_t.max()+RATE_BIN, RATE_BIN)
hit_cnt,_ = np.histogram(ae_t, bins=t_bins_r)
hit_rate  = hit_cnt / RATE_BIN          # hits/s per bin

# 早期基准速率：取前 T_EARLY 秒内各 bin 速率的中位数
early_bins = t_bins_r[:-1] < T_EARLY
baseline_rate = float(np.median(hit_rate[early_bins])) if early_bins.any() else 1.0
baseline_rate = max(baseline_rate, 0.1)  # 防止零值
burst_thresh  = baseline_rate * BURST_FACTOR

# 将每个 hit 的 bin 速率赋回
bin_idx    = np.clip(np.searchsorted(t_bins_r[1:], ae_t), 0, len(hit_cnt)-1)
rate_at_t  = hit_rate[bin_idx]
is_burst   = rate_at_t > burst_thresh

n_burst_hits = int(is_burst.sum())
n_burst_bins = int((hit_rate > burst_thresh).sum())
print(f"  基准速率: {baseline_rate:.2f} hits/s  爆发阈值: {burst_thresh:.2f} hits/s")
print(f"  爆发段 bins: {n_burst_bins}/{len(hit_cnt)}  爆发段 hits: {n_burst_hits} ({100.*n_burst_hits/n_hits:.1f}%)")

# ── Step-3  组合干扰判定 ───────────────────────────────────────────────
# 非爆发段：时间窗 + 振幅带（干扰主导）
# 爆发段：仅时间窗（真实AE主导，振幅带删除暂停）
contam_amp_quiet = in_amp_band & ~is_burst   # 非爆发段内振幅带撞击
contam = contam_tw | contam_amp_quiet

ae_clean  = ae[~contam].reset_index(drop=True)
ae_contam = ae[ contam].reset_index(drop=True)
n_contam  = int(contam.sum()); n_clean = n_hits-n_contam; pct = 100.*n_contam/n_hits

# 各步贡献
n_tw_only   = int((contam_tw & ~contam_amp_quiet).sum())
n_amp_quiet = int(contam_amp_quiet.sum())
n_amp_burst = int((in_amp_band & is_burst).sum())   # 爆发段内振幅带撞击（被保留为真实AE）

print(f"  时间窗去除:         {n_tw_only} hits ({100.*n_tw_only/n_hits:.1f}%)")
print(f"  振幅带(非爆发)去除: {n_amp_quiet} hits ({100.*n_amp_quiet/n_hits:.1f}%)")
print(f"  振幅带(爆发内保留): {n_amp_burst} hits ({100.*n_amp_burst/n_hits:.1f}%) ← 真实AE")
print(f"  总干扰: {n_contam} ({pct:.1f}%)  真实AE: {n_clean} ({100-pct:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# § 4  绘图
# ═══════════════════════════════════════════════════════════════════════════
print("\n生成图表...")

def vfail(axes,tf):
    if not np.isnan(tf):
        for ax in axes: ax.axvline(tf,color='red',ls='--',lw=1.2,alpha=0.7)

# 爆发段时间范围（用于图中背景色）
burst_bin_times = t_bins_r[:-1][hit_rate>burst_thresh]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图1  爆发速率 + 振幅带诊断
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig1, axes1 = plt.subplots(2,2,figsize=(18,10))
fig1.suptitle(f'v6 干扰识别诊断  振幅带[{AMP_BAND_LOW:.1f},{AMP_BAND_HIGH:.1f}]dB  '
              f'爆发阈值={burst_thresh:.1f} hits/s',fontsize=12,fontweight='bold')

# 全段撞击速率
ax=axes1[0,0]
tc = (t_bins_r[:-1]+t_bins_r[1:])/2
ax.plot(tc, hit_rate, color='steelblue', lw=0.8, label='撞击速率')
ax.axhline(burst_thresh, color='red', lw=1.5, ls='--', label=f'爆发阈值 {burst_thresh:.1f} hits/s')
ax.axhline(baseline_rate, color='gray', lw=1, ls=':', label=f'基准速率 {baseline_rate:.2f} hits/s')
for bt in burst_bin_times:
    ax.axvspan(bt, bt+RATE_BIN, alpha=0.15, color='orange', zorder=0)
ax.set_yscale('log'); ax.set_xlabel('时间 (s)'); ax.set_ylabel('撞击速率 (hits/s)')
ax.set_title('全段撞击速率（橙色=爆发段）'); ax.legend(fontsize=8)
vfail([ax],t_fail)

# 振幅带拟合
ax=axes1[0,1]
ax.bar(ac_fine, hist_fit, width=0.9, color='tomato', alpha=0.6, label=f'前{T_EARLY:.0f}s时间窗内确认干扰')
ax.plot(ac_fine, hs, color='darkred', lw=2, label='高斯平滑')
ax.axvspan(AMP_BAND_LOW,AMP_BAND_HIGH,alpha=0.15,color='red')
ax.axvline(AMP_BAND_LOW,color='red',lw=1.5,ls='--',label=f'干扰带 [{AMP_BAND_LOW:.0f},{AMP_BAND_HIGH:.0f}]dB')
ax.axvline(AMP_BAND_HIGH,color='red',lw=1.5,ls='--')
ax.axvline(amp_peak,color='darkred',lw=2,label=f'峰值 {amp_peak:.1f} dB')
ax.set_xlabel('振幅 (dB)'); ax.set_ylabel('频数')
ax.set_title('干扰振幅带拟合'); ax.legend(fontsize=8)

# 时间 vs 振幅散点（按类别着色）
ax=axes1[1,0]
mask_clean_quiet  = ~contam & ~is_burst
mask_clean_burst  = ~contam & is_burst
mask_contam_amp   = contam_amp_quiet
mask_contam_tw    = contam_tw & ~contam_amp_quiet
ax.scatter(ae_t[mask_contam_tw],   ae_amp[mask_contam_tw],   s=1,alpha=0.3,color='silver',  label='时间窗干扰')
ax.scatter(ae_t[mask_contam_amp],  ae_amp[mask_contam_amp],  s=1,alpha=0.3,color='tomato',  label='振幅带干扰')
ax.scatter(ae_t[mask_clean_quiet], ae_amp[mask_clean_quiet], s=1.5,alpha=0.5,color='steelblue',label='真实AE(非爆发)')
ax.scatter(ae_t[mask_clean_burst], ae_amp[mask_clean_burst], s=1.5,alpha=0.6,color='green',  label='真实AE(爆发段)')
ax.axhspan(AMP_BAND_LOW,AMP_BAND_HIGH,alpha=0.06,color='red')
ax.axhline(AMP_BAND_LOW, color='red',lw=0.7,ls='--',alpha=0.6)
ax.axhline(AMP_BAND_HIGH,color='red',lw=0.7,ls='--',alpha=0.6)
ax.set_xlabel('时间 (s)'); ax.set_ylabel('振幅 (dB)')
ax.set_title('全段散点（类别着色）')
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0],marker='o',ls='None',color='silver',  markersize=4,label=f'时间窗干扰 ({n_tw_only})'),
    Line2D([0],[0],marker='o',ls='None',color='tomato',  markersize=4,label=f'振幅带干扰(非爆发) ({n_amp_quiet})'),
    Line2D([0],[0],marker='o',ls='None',color='steelblue',markersize=4,label=f'真实AE(非爆发) ({int(mask_clean_quiet.sum())})'),
    Line2D([0],[0],marker='o',ls='None',color='green',   markersize=4,label=f'真实AE(爆发段) ({int(mask_clean_burst.sum())})'),
],fontsize=7,loc='upper left')
vfail([ax],t_fail)

# 振幅分布对比
ax=axes1[1,1]
ae2=np.arange(35,105,2); ac2=(ae2[:-1]+ae2[1:])/2
h_tw,_  = np.histogram(ae_amp[contam_tw&~contam_amp_quiet],bins=ae2)
h_aq,_  = np.histogram(ae_amp[contam_amp_quiet],bins=ae2)
h_cl,_  = np.histogram(ae_amp[~contam],bins=ae2)
ax.bar(ac2,h_tw,width=1.8,color='silver',  alpha=0.8,label=f'时间窗干扰 ({n_tw_only})')
ax.bar(ac2,h_aq,width=1.8,color='tomato',  alpha=0.8,bottom=h_tw,label=f'振幅带干扰(非爆发) ({n_amp_quiet})')
ax.bar(ac2,h_cl,width=1.8,color='steelblue',alpha=0.7,bottom=h_tw+h_aq,label=f'真实AE ({n_clean})')
ax.axvspan(AMP_BAND_LOW,AMP_BAND_HIGH,alpha=0.10,color='red')
ax.axvline(AMP_BAND_LOW, color='red',lw=1.2,ls='--')
ax.axvline(AMP_BAND_HIGH,color='red',lw=1.2,ls='--')
ax.set_xlabel('振幅 (dB)'); ax.set_ylabel('频数')
ax.set_title('振幅分布（堆叠）'); ax.legend(fontsize=8)

plt.tight_layout()
out1=os.path.join(RESULT_DIR,'v6_01_干扰识别诊断.png')
fig1.savefig(out1); plt.close(fig1)
print(f"图1已保存: {out1}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图2  全段干扰对比（6通道）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("生成全段对比图（6通道）...")
T_MAX=ae_t.max()+20; SZ,AL=1.5,0.40; YL=[ae_amp.min()-3,ae_amp.max()+3]

fig2=plt.figure(figsize=(20,28))
gs2=gridspec.GridSpec(6,2,figure=fig2,hspace=0.10,wspace=0.06,
                      top=0.93,bottom=0.04,left=0.07,right=0.97)
fig2.suptitle(
    f'声发射振幅 全段干扰对比 v6 (0-{T_MAX:.0f}s)\n'
    f'非爆发段振幅带[{AMP_BAND_LOW:.1f},{AMP_BAND_HIGH:.1f}]dB全段删除 + 时间窗补充\n'
    f'爆发段（撞击速率>{burst_thresh:.1f}hits/s）振幅带删除暂停，保留真实AE\n'
    f'总干扰 {n_contam} ({pct:.1f}%)  真实AE {n_clean} ({100-pct:.1f}%)',
    fontsize=10,fontweight='bold')

for ch in range(1,7):
    row=ch-1
    d_ct=ae_contam[ae_contam['CH']==ch]; d_cl=ae_clean[ae_clean['CH']==ch]

    axL=fig2.add_subplot(gs2[row,0])
    if len(d_ct): axL.scatter(d_ct['Time'],d_ct['AMP'],s=SZ,alpha=AL*0.5,color='silver',zorder=1,rasterized=True)
    if len(d_cl): axL.scatter(d_cl['Time'],d_cl['AMP'],s=SZ,alpha=AL,color=CH_COLORS[ch],zorder=2,rasterized=True)
    # 爆发段背景色
    for bt in burst_bin_times:
        axL.axvspan(bt,bt+RATE_BIN,alpha=0.06,color='limegreen',zorder=0)
    axL.axhspan(AMP_BAND_LOW,AMP_BAND_HIGH,alpha=0.08,color='red',zorder=0)
    axL.axhline(AMP_BAND_LOW, color='red',lw=0.7,ls='--',alpha=0.6)
    axL.axhline(AMP_BAND_HIGH,color='red',lw=0.7,ls='--',alpha=0.6)
    axL.set_ylabel(f'CH{ch}\n振幅(dB)',fontsize=9)
    axL.set_ylim(YL); axL.set_xlim(0,T_MAX)
    axL.yaxis.set_major_locator(MultipleLocator(20))
    axL.yaxis.set_minor_locator(MultipleLocator(10))
    if row==0:
        axL.set_title(f'原始 ({n_hits} hits)',fontsize=11,pad=8)
        axL.legend(handles=[
            Line2D([0],[0],marker='o',ls='None',color='silver',markersize=4,label=f'干扰 ({n_contam})'),
            Line2D([0],[0],marker='o',ls='None',color=CH_COLORS[ch],markersize=4,label=f'真实AE ({n_clean})'),
            Line2D([0],[0],color='limegreen',lw=4,alpha=0.4,label='爆发段（振幅带保留）')],
            fontsize=7,loc='upper left')

    axR=fig2.add_subplot(gs2[row,1],sharey=axL)
    if len(d_cl): axR.scatter(d_cl['Time'],d_cl['AMP'],s=SZ,alpha=AL,color=CH_COLORS[ch],rasterized=True)
    for bt in burst_bin_times:
        axR.axvspan(bt,bt+RATE_BIN,alpha=0.08,color='limegreen',zorder=0)
    axR.axhspan(AMP_BAND_LOW,AMP_BAND_HIGH,alpha=0.05,color='red',zorder=0)
    axR.axhline(AMP_BAND_LOW, color='red',lw=0.6,ls='--',alpha=0.4)
    axR.axhline(AMP_BAND_HIGH,color='red',lw=0.6,ls='--',alpha=0.4)
    axR.set_xlim(0,T_MAX); axR.set_ylim(YL)
    axR.yaxis.set_major_locator(MultipleLocator(20)); axR.tick_params(labelleft=False)
    if row==0: axR.set_title(f'去干扰后 ({n_clean} hits)',fontsize=11,pad=8)
    if row<5:
        axL.tick_params(labelbottom=False); axR.tick_params(labelbottom=False)
    else:
        axL.set_xlabel('时间 (s)'); axR.set_xlabel('时间 (s)')
    if not np.isnan(t_fail):
        axL.axvline(t_fail,color='red',lw=0.8,ls=':',alpha=0.6)
        axR.axvline(t_fail,color='red',lw=0.8,ls=':',alpha=0.6)

out2=os.path.join(RESULT_DIR,'v6_02_全段干扰对比图.png')
fig2.savefig(out2); plt.close(fig2)
print(f"图2已保存: {out2}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图3  Vp + 综合分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig3,axes3=plt.subplots(4,1,figsize=(16,18),sharex=True)
fig3.suptitle('超声波与声发射综合分析 v6\n'
              'Vp(AIC calibrated) + AE(burst-protected amplitude-band removal)',
              fontsize=12,fontweight='bold')
ax3a,ax3b,ax3c,ax3d=axes3

mv=~np.isnan(us_Vp_aic)
ax3a.plot(us_ts[mv],us_Vp_aic[mv],color='crimson',lw=1,alpha=0.85,
          label=f'Vp AIC (t_cal={sys_delay:.2f}us)')
ax3a.fill_between(us_ts[mv],us_Vp_aic[mv],alpha=0.12,color='crimson')
ax3a.set_ylabel('Vp (km/s)')
ax3a.set_title(f'P波速度  初始{np.nanmean(us_Vp_aic[valid_aic][:10]):.2f} km/s  峰值{np.nanmax(us_Vp_aic):.2f} km/s')
ax3a.yaxis.set_minor_locator(AutoMinorLocator()); ax3a.legend(loc='upper left')

for ch in range(1,7):
    d=ae_clean[ae_clean['CH']==ch]
    ax3b.scatter(d['Time'],d['AMP'],s=1.5,alpha=0.35,color=CH_COLORS[ch],rasterized=True)
# 爆发段背景
for bt in burst_bin_times:
    ax3b.axvspan(bt,bt+RATE_BIN,alpha=0.06,color='limegreen',zorder=0)
ax3b.axhspan(AMP_BAND_LOW,AMP_BAND_HIGH,alpha=0.06,color='red')
ax3b.axhline(AMP_BAND_LOW, color='red',lw=0.8,ls=':',alpha=0.5)
ax3b.axhline(AMP_BAND_HIGH,color='red',lw=0.8,ls=':',alpha=0.5,
             label=f'已剔除带[{AMP_BAND_LOW:.0f},{AMP_BAND_HIGH:.0f}]dB\n(爆发段内除外)')
ax3b.set_ylabel('AE振幅 (dB)'); ax3b.set_title('声发射振幅 (v6 去干扰后)')
ax3b.legend(handles=[plt.Line2D([0],[0],marker='o',ls='None',color=CH_COLORS[i+1],
            markersize=5,label=f'CH{i+1}') for i in range(6)],
            ncol=6,fontsize=8,loc='upper left',framealpha=0.6)

ax3c.scatter(ae_clean['Time'],ae_clean['ABS_E'],s=1.5,alpha=0.3,color='darkred',rasterized=True)
ax3c.set_yscale('log'); ax3c.set_ylabel('绝对能量 (aJ)')
ax3c.set_title('声发射绝对能量 (去干扰后)')

ax3d.plot(ae['Time'],np.arange(1,n_hits+1),color='tomato',lw=1.2,ls='--',alpha=0.7,
          label=f'原始 ({n_hits})')
ax3d.plot(ae_clean.sort_values('Time')['Time'],np.arange(1,n_clean+1),
          color='steelblue',lw=1.5,alpha=0.9,label=f'去干扰后 ({n_clean})')
ax3d.set_ylabel('累计撞击数'); ax3d.set_xlabel('时间 (s)')
ax3d.set_title('累计撞击数对比'); ax3d.legend()

vfail(axes3,t_fail)
if not np.isnan(t_fail):
    axes3[0].text(t_fail+15,ax3a.get_ylim()[0]*1.02,f'破坏 {t_fail:.0f}s',color='red',fontsize=9)

plt.tight_layout()
out3=os.path.join(RESULT_DIR,'v6_03_综合分析.png')
fig3.savefig(out3); plt.close(fig3)
print(f"图3已保存: {out3}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图4  Vp 演化详图
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig4,axes4=plt.subplots(2,2,figsize=(16,10),sharex='col')
fig4.suptitle(f'P波速度演化 (对零AIC校正)  t_cal={sys_delay:.3f} us',fontsize=12,fontweight='bold')
(ax4a,ax4b),(ax4c,ax4d)=axes4
ax4a.plot(us_ts,us_pt_sw, color='navy',  lw=0.8,alpha=0.8,label='软件')
ax4a.plot(us_ts,us_pt_aic,color='crimson',lw=0.8,alpha=0.8,label='AIC')
ax4a.set_ylabel('P波到时 (us)'); ax4a.set_title('P波到时'); ax4a.legend()
ax4b.plot(us_ts,us_pt_aic-us_pt_sw,color='purple',lw=0.6,alpha=0.7)
ax4b.axhline(np.nanmean(diff_valid),color='red',lw=1,ls=':',label=f'均值{np.nanmean(diff_valid):.2f}us')
ax4b.set_ylabel('AIC-软件 (us)'); ax4b.set_title('拾取差值'); ax4b.legend()
vb=~(np.isnan(us_Vp_aic)|np.isnan(us_Vp_sw))
ax4c.plot(us_ts[vb],us_Vp_sw[vb], color='navy',  lw=0.8,alpha=0.8,label='Vp(软件)')
ax4c.plot(us_ts[vb],us_Vp_aic[vb],color='crimson',lw=0.8,alpha=0.8,label='Vp(AIC)')
ax4c.set_ylabel('Vp (km/s)'); ax4c.set_xlabel('时间 (s)')
ax4c.set_title(f'P波速度  {H_MM:.0f}mm/(t_AIC-{sys_delay:.2f}us)')
ax4c.legend(); ax4c.yaxis.set_minor_locator(AutoMinorLocator())
lim=[min(np.nanmin(us_Vp_sw[vb]),np.nanmin(us_Vp_aic[vb]))*0.95,
     max(np.nanmax(us_Vp_sw[vb]),np.nanmax(us_Vp_aic[vb]))*1.05]
ax4d.scatter(us_Vp_sw[vb],us_Vp_aic[vb],s=3,alpha=0.4,color='steelblue')
ax4d.plot(lim,lim,'r--',lw=1,label='1:1')
ax4d.set_xlabel('Vp软件'); ax4d.set_ylabel('Vp AIC'); ax4d.legend()
ax4d.set_xlim(lim); ax4d.set_ylim(lim); ax4d.set_title('散点对比')
vfail([ax4a,ax4c],t_fail)
plt.tight_layout()
out4=os.path.join(RESULT_DIR,'v6_04_P波速度.png')
fig4.savefig(out4); plt.close(fig4)
print(f"图4已保存: {out4}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图5  AE 事件空间分布
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
out5=None
if len(evts)>0 and 'x' in evts.columns:
    fig5,axes5=plt.subplots(1,3,figsize=(18,7))
    fig5.suptitle(f'声发射事件空间分布 ({len(evts)} 事件)',fontsize=12,fontweight='bold')
    sk=dict(c=evts['time'],cmap='plasma',s=4,alpha=0.5)
    axes5[0].scatter(evts['x'],evts['y'],**sk); axes5[0].set_xlabel('x(mm)'); axes5[0].set_ylabel('y(mm)'); axes5[0].set_title('XY')
    sc=axes5[1].scatter(evts['x'],evts['z'],**sk); axes5[1].set_xlabel('x(mm)'); axes5[1].set_ylabel('z(mm)'); axes5[1].set_title('XZ')
    axes5[2].scatter(evts['y'],evts['z'],**sk); axes5[2].set_xlabel('y(mm)'); axes5[2].set_ylabel('z(mm)'); axes5[2].set_title('YZ')
    plt.colorbar(sc,ax=axes5[2],label='时间 (s)',shrink=0.8)
    plt.tight_layout()
    out5=os.path.join(RESULT_DIR,'v6_05_AE事件空间分布.png')
    fig5.savefig(out5); plt.close(fig5)
    print(f"图5已保存: {out5}")

# ─── CSV ─────────────────────────────────────────────────────────────────
out_vp=os.path.join(RESULT_DIR,'v6_Vp_AIC.csv')
pd.DataFrame({'time_s':us_ts,'pwave_sw_us':us_pt_sw,'pwave_aic_us':us_pt_aic,
              'travel_aic_us':us_pt_aic-sys_delay,'Vp_sw_km':us_Vp_sw,
              'Vp_aic_km':us_Vp_aic}).to_csv(out_vp,index=False)
out_ae=os.path.join(RESULT_DIR,'v6_AE_clean.csv'); ae_clean.to_csv(out_ae,index=False)
out_ct=os.path.join(RESULT_DIR,'v6_AE_contaminated.csv'); ae_contam.to_csv(out_ct,index=False)

# ─── 汇总 ─────────────────────────────────────────────────────────────────
print("\n"+"="*60)
print("[超声波 Vp]")
print(f"  t_cal={sys_delay:.3f}us  初始Vp={np.nanmean(us_Vp_aic[valid_aic][:10]):.3f}km/s  峰值Vp={np.nanmax(us_Vp_aic):.3f}km/s")
if not np.isnan(t_fail):
    pfi=np.where(fail_mask)[0][0]
    print(f"  破坏前5次均值={np.nanmean(us_Vp_aic[max(0,pfi-5):pfi]):.3f}km/s  破坏时刻={t_fail:.1f}s")
print(f"\n[声发射干扰识别 v6]")
print(f"  振幅带: [{AMP_BAND_LOW:.1f},{AMP_BAND_HIGH:.1f}] dB (峰值 {amp_peak:.1f} dB)")
print(f"  基准速率: {baseline_rate:.2f} hits/s  爆发阈值: {burst_thresh:.2f} hits/s")
print(f"  爆发段 hits (振幅带内已保留): {n_amp_burst} ({100.*n_amp_burst/n_hits:.1f}%)")
print(f"  时间窗去除:       {n_tw_only} hits ({100.*n_tw_only/n_hits:.1f}%)")
print(f"  振幅带(非爆发)去除:{n_amp_quiet} hits ({100.*n_amp_quiet/n_hits:.1f}%)")
print(f"  总干扰: {n_contam} ({pct:.1f}%)  真实AE: {n_clean} ({100-pct:.1f}%)")
print(f"\n[输出目录] {RESULT_DIR}")
for f in [out1,out2,out3,out4,out_vp,out_ae,out_ct]+([out5] if out5 else []):
    print(f"  {os.path.basename(f)}")
print("\n分析完成！")
