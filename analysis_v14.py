#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花岗岩单轴压缩试验 - 综合分析 v14
干扰识别：时变振幅条带跟踪法
  原理：
    1. 以 US 脉冲时间窗内的 hits 为参考，逐时间段（T_BIN=50s）拟合
       每个通道干扰条带的中心振幅（取时间窗内振幅分布的峰值）
    2. 对拟合结果做高斯平滑，得到随时间变化的干扰中心曲线
    3. 删除所有落在"中心±STRIPE_BW/2"范围内的 hits（与时间无关）
    4. CH4 不过滤
  优点：
    - 无需固定振幅带，自动跟踪干扰中心随压缩状态的漂移
    - 近破坏段振幅条带上升，曲线同步上升，不会误删带外的真实 AE
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
from matplotlib.lines import Line2D
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
BASE      = r'g:\Cursor project\ZCY-shengfashe'
US_FILE   = os.path.join(BASE,'超声波','04-15 - ultrasonics data.csv')
CAL_FILE  = os.path.join(BASE,'超声波','chushi.csv')
AE_HITS   = os.path.join(BASE,'声发射','04-15-hits-振铃计数、能量等.TXT')
AE_EVTS   = os.path.join(BASE,'声发射','04-15-声发射事件.TXT')
RESULT_DIR= os.path.join(BASE,'结果')
os.makedirs(RESULT_DIR, exist_ok=True)

# ─── 参数 ─────────────────────────────────────────────────────────────────
H_MM = 100.0;  H_M = H_MM/1000.0
FS_HZ = 40e6
BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER = 50e3, 700e3, 4
AIC_SEARCH_OFFSET_US = 12.0
AIC_SEARCH_WIDTH_US  = 30.0
AIC_GLOBAL_START_US  = 5.0

# ── 干扰条带跟踪参数 ───────────────────────────────────────────────────────
US_MASK_POST_REF = 0.50   # 仅用于识别参考 hits 的时间窗（内部使用，非最终过滤）
US_MASK_PRE_REF  = 0.05

T_BIN      = 50.0   # 时间分箱宽度 (s)，用于逐段拟合干扰中心振幅
SIGMA_BINS = 3      # 干扰中心曲线高斯平滑宽度（bins 数）
MIN_REF_N  = 3      # 每个时间箱最少参考 hits 数，不足则插值

# 各通道条带半宽（dB）：中心曲线 ± 此值内的 hits 均视为干扰
# 可逐通道调整；越大 → 删得越多；越小 → 保留越多
CH_STRIPE_HW = {
    1: 6,    # ±6 dB → 共 12 dB
    2: 7,    # ±7 dB → 共 14 dB（CH2 干扰略宽）
    3: 5,    # ±5 dB → 共 10 dB
    4: 0,    # CH4 不过滤
    5: 8,    # ±8 dB → 共 16 dB（CH5 干扰较宽）
    6: 6,    # ±6 dB → 共 12 dB
}

CH_NO_FILTER = {4}   # 无干扰通道，全部保留

RATE_BIN    = 5.0    # 速率分箱宽度 (s)
RATE_SMOOTH = 3      # 速率平滑窗

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
# § 1  超声波 AIC + Vp
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
print(f"  {n_sw} sweeps  {us_ts[0]:.1f}-{us_ts[-1]:.1f} s")

sos_bp = bp_sos(FS_HZ,BP_LOW_HZ,BP_HIGH_HZ,BP_ORDER)

print("对零校准...")
t_cal_aic=np.nan
try:
    cal=pd.read_csv(CAL_FILE,header=None,skiprows=154,low_memory=False,dtype=str,encoding='latin-1')
    ct=pd.to_numeric(cal.iloc[:,0],errors='coerce').values; cs_=pd.to_numeric(cal.iloc[:,1],errors='coerce').values
    ok=~(np.isnan(ct)|np.isnan(cs_)); ct,cs_=ct[ok],cs_[ok]
    snr=np.max(np.abs(cs_))/(np.std(cs_[:100])+1e-30)
    if snr>10: t_cal_aic,_,_=aic_pick(apply_bp(cs_,sos_bp),ct,0.5,40.0)
    print(f"  SNR={snr:.0f}  t_cal={t_cal_aic:.3f} us")
except Exception as e:
    print(f"  警告: {e}")
if np.isnan(t_cal_aic):
    es=us_pt_sw[~np.isnan(us_pt_sw)][:20]
    t_cal_aic=float(np.nanmedian(es)-H_M/4800.0*1e6) if len(es) else 0.0
sys_delay=t_cal_aic

print("AIC 拾取...")
t_arr=wf_time_us.copy(); us_pt_aic=np.full(n_sw,np.nan)
for i in range(n_sw):
    wf=wf_data[:,i].astype(float)
    if np.sum(np.isnan(wf))>n_samp*0.5: continue
    wf[np.isnan(wf)]=0.0
    ref=us_pt_sw[i] if not np.isnan(us_pt_sw[i]) else 30.0
    us_pt_aic[i],_,_=aic_pick(apply_bp(wf,sos_bp),t_arr,
                               max(AIC_GLOBAL_START_US,ref-AIC_SEARCH_OFFSET_US),
                               ref+AIC_SEARCH_WIDTH_US)
valid_aic=~np.isnan(us_pt_aic)
tr_aic=(us_pt_aic-sys_delay)*1e-6; tr_sw=(us_pt_sw-sys_delay)*1e-6
us_Vp_aic=np.where((tr_aic>5e-6)&(tr_aic<200e-6),H_M/tr_aic/1000.,np.nan)
us_Vp_sw =np.where((tr_sw >5e-6)&(tr_sw <200e-6),H_M/tr_sw /1000.,np.nan)
diff_valid=(us_pt_aic-us_pt_sw)[valid_aic&~np.isnan(us_pt_sw)]
print(f"  Vp(AIC): {np.nanmin(us_Vp_aic):.2f}-{np.nanmax(us_Vp_aic):.2f} km/s")

vp_base=np.nanmedian(us_Vp_aic[valid_aic][:30]) if valid_aic.sum()>=30 else np.nan
fail_mask=(~np.isnan(us_Vp_aic))&(us_Vp_aic<vp_base*0.7) if not np.isnan(vp_base) else np.zeros(n_sw,bool)
t_fail=float(us_ts[np.where(fail_mask)[0][0]]) if np.any(fail_mask) else np.nan
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

ae   =parse_ae_hits(AE_HITS); ae=ae[ae['Time']>0].sort_values('Time').reset_index(drop=True)
ae_t =ae['Time'].values; ae_amp=ae['AMP'].values; ae_ch=ae['CH'].values; n_hits=len(ae)
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

evts=parse_ae_events(AE_EVTS)
print(f"  {len(evts)} 已定位事件")


# ═══════════════════════════════════════════════════════════════════════════
# § 3  干扰识别 v14 — 时变振幅条带跟踪
# ═══════════════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("干扰识别 v14（时变振幅条带跟踪法）...")

us_ts_s = np.sort(us_ts)
period_approx = float(np.median(np.diff(us_ts_s)))

# ── 参考 hits：时间窗内（仅用于拟合干扰中心，不直接用于过滤）──────────────
_idx      = np.searchsorted(us_ts_s, ae_t, side='right') - 1
dt_after  = np.where(_idx>=0, ae_t - us_ts_s[np.maximum(_idx,0)], np.nan)
_idx_n    = np.searchsorted(us_ts_s, ae_t, side='left')
dt_before = np.where(_idx_n<len(us_ts_s), us_ts_s[np.minimum(_idx_n,len(us_ts_s)-1)] - ae_t, np.inf)
in_tw_ref = ((dt_after>=0)&(dt_after<=US_MASK_POST_REF)) | \
            ((dt_before>=0)&(dt_before<=US_MASK_PRE_REF))

# ── 逐通道拟合干扰中心振幅曲线 ────────────────────────────────────────────
ae_fine = np.arange(35,106,1); ac_fine = (ae_fine[:-1]+ae_fine[1:])/2
t_bins_fit  = np.arange(0, ae_t.max()+T_BIN, T_BIN)
t_centers_fit = (t_bins_fit[:-1]+t_bins_fit[1:])/2
n_fit_bins  = len(t_centers_fit)

ch_center_curve = {}   # ch -> (t_arr, amp_arr)   用于绘图
interf_center   = np.full(n_hits, np.nan)  # 每个 hit 处的干扰中心振幅

print(f"  T_BIN={T_BIN}s  SIGMA_BINS={SIGMA_BINS}  参考窗={US_MASK_POST_REF}s")
print(f"  {'CH':>3}  {'半宽(dB)':>8}  {'参考hits':>8}  {'有效bins':>8}  {'中心范围(dB)':>16}")

for ch in range(1,7):
    hw = CH_STRIPE_HW.get(ch, 6)
    if ch in CH_NO_FILTER or hw == 0:
        print(f"  CH{ch}  {'---':>8}  {'---':>8}  {'---':>8}  不过滤")
        continue

    ch_mask = ae_ch == ch
    ref_mask = in_tw_ref & ch_mask
    t_ref  = ae_t[ref_mask]
    amp_ref= ae_amp[ref_mask]

    # 逐时间箱求振幅分布峰值（更稳健于 median，抗少量真实AE混入）
    bin_center = np.full(n_fit_bins, np.nan)
    for i in range(n_fit_bins):
        sel = (t_ref>=t_bins_fit[i]) & (t_ref<t_bins_fit[i+1])
        if sel.sum() >= MIN_REF_N:
            h,_ = np.histogram(amp_ref[sel], bins=ae_fine)
            hs  = gaussian_filter1d(h.astype(float), sigma=1)
            pk  = int(np.argmax(hs))
            bin_center[i] = ac_fine[pk]

    # 插值 + 平滑
    valid = ~np.isnan(bin_center)
    n_valid = valid.sum()
    if n_valid < 2:
        print(f"  CH{ch}  {hw:>8}  {ref_mask.sum():>8}  {n_valid:>8}  参考不足，跳过")
        continue
    bc_filled = bin_center.copy()
    vx = t_centers_fit[valid]; vy = bin_center[valid]
    # 边缘外推，中间线性插值
    bc_filled = np.interp(t_centers_fit, vx, vy)
    bc_smooth = gaussian_filter1d(bc_filled, sigma=SIGMA_BINS)

    ch_center_curve[ch] = (t_centers_fit.copy(), bc_smooth.copy())

    # 插值到该通道所有 hits
    interf_center[ch_mask] = np.interp(ae_t[ch_mask], t_centers_fit, bc_smooth)

    cmin, cmax = bc_smooth.min(), bc_smooth.max()
    print(f"  CH{ch}  {hw:>8}  {ref_mask.sum():>8}  {n_valid:>8}  [{cmin:.1f},{cmax:.1f}] dB")

# ── 干扰判定：落在时变条带内 ──────────────────────────────────────────────
no_filter = np.isin(ae_ch, list(CH_NO_FILTER))
on_stripe = np.zeros(n_hits, dtype=bool)
for ch in range(1,7):
    hw = CH_STRIPE_HW.get(ch,6)
    if ch in CH_NO_FILTER or hw==0: continue
    cm = ae_ch==ch
    has_center = ~np.isnan(interf_center)
    on_stripe |= cm & has_center & (np.abs(ae_amp - interf_center) <= hw)

contam    = on_stripe & ~no_filter
ae_clean  = ae[~contam].reset_index(drop=True)
ae_contam = ae[ contam].reset_index(drop=True)
n_contam  = int(contam.sum()); n_clean=n_hits-n_contam; pct=100.*n_contam/n_hits

print(f"\n  时变条带内(干扰): {n_contam} ({pct:.1f}%)")
print(f"  条带外(保留):     {n_clean} ({100-pct:.1f}%)")
for ch in range(1,7):
    cm=ae_ch==ch
    print(f"    CH{ch}: 删 {int((contam&cm).sum())}  留 {int((~contam&cm).sum())}"
          +(' ← 不过滤' if ch in CH_NO_FILTER else ''))

# ── 爆发段检测（仅用于可视化） ─────────────────────────────────────────────
AMP_HIGH_THRESH=74.0
above_mask=ae_amp>AMP_HIGH_THRESH
t_bins_r=np.arange(0,ae_t.max()+RATE_BIN,RATE_BIN)
hit_cnt_ab,_=np.histogram(ae_t[above_mask],bins=t_bins_r)
hit_rate_ab=(hit_cnt_ab/RATE_BIN).astype(float)
kernel=np.ones(RATE_SMOOTH)/RATE_SMOOTH
hit_rate_ab_sm=np.convolve(hit_rate_ab,kernel,mode='same')
T_EARLY=200.0
early_b=t_bins_r[:-1]<T_EARLY
baseline_ab=max(float(np.median(hit_rate_ab_sm[early_b])),0.005) if early_b.any() else 0.01
burst_thresh=max(baseline_ab*3.0,0.05)
is_burst_bin=hit_rate_ab_sm>burst_thresh
bin_idx=np.clip(np.searchsorted(t_bins_r[1:],ae_t),0,len(hit_cnt_ab)-1)
is_burst=is_burst_bin[bin_idx]
t_burst_start=float(t_bins_r[np.where(is_burst_bin)[0][0]]) if is_burst_bin.any() else np.nan
burst_bin_times=t_bins_r[:-1][is_burst_bin]
print(f"  爆发段参考: ~{t_burst_start:.0f}s 起")


# ═══════════════════════════════════════════════════════════════════════════
# § 4  绘图
# ═══════════════════════════════════════════════════════════════════════════
print("\n生成图表...")

def vfail(axes,tf):
    if not np.isnan(tf):
        for ax in axes: ax.axvline(tf,color='red',ls='--',lw=1.2,alpha=0.7)

T_MAX=ae_t.max()+20; SZ,AL=1.5,0.40; YL=[ae_amp.min()-3,ae_amp.max()+3]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图1  各通道干扰中心曲线拟合诊断
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig1=plt.figure(figsize=(20,14))
gs1=gridspec.GridSpec(3,4,figure=fig1,hspace=0.45,wspace=0.35,top=0.93,bottom=0.06)
fig1.suptitle(
    f'v14 干扰识别诊断  时变振幅条带跟踪\n'
    f'参考窗={US_MASK_POST_REF}s  T_BIN={T_BIN}s  σ={SIGMA_BINS}bins  '
    f'干扰 {n_contam} ({pct:.1f}%)  真实AE {n_clean} ({100-pct:.1f}%)',
    fontsize=10,fontweight='bold')

for ch in range(1,7):
    row_idx,col_idx=(0,ch-1) if ch<=4 else (1,ch-5)
    ax=fig1.add_subplot(gs1[row_idx,col_idx])
    hw=CH_STRIPE_HW.get(ch,6)
    cm=ae_ch==ch

    # 散点：参考 hits（窗内）
    ref_ch = in_tw_ref & cm
    ax.scatter(ae_t[ref_ch], ae_amp[ref_ch], s=1, alpha=0.4,
               color='tomato', zorder=2, rasterized=True, label=f'参考(窗内) n={ref_ch.sum()}')
    # 散点：窗外 hits
    out_ch = ~in_tw_ref & cm
    ax.scatter(ae_t[out_ch], ae_amp[out_ch], s=1, alpha=0.15,
               color='gray', zorder=1, rasterized=True)

    # 干扰中心曲线 + 条带范围
    if ch in ch_center_curve:
        tc_arr, bc_arr = ch_center_curve[ch]
        ax.plot(tc_arr, bc_arr,       color='navy', lw=1.5, zorder=4, label='干扰中心')
        ax.plot(tc_arr, bc_arr+hw,    color='navy', lw=0.8, ls='--', zorder=4, alpha=0.7)
        ax.plot(tc_arr, bc_arr-hw,    color='navy', lw=0.8, ls='--', zorder=4, alpha=0.7)
        ax.fill_between(tc_arr, bc_arr-hw, bc_arr+hw, alpha=0.12, color='navy', zorder=3)

    ax.set_xlim(0,T_MAX); ax.set_ylim(YL)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('振幅 (dB)')
    note='(不过滤)' if ch in CH_NO_FILTER else f'±{hw}dB  删{int((contam&cm).sum())} 留{int((~contam&cm).sum())}'
    ax.set_title(f'CH{ch}  {note}')
    ax.legend(fontsize=6, loc='upper left')
    if not np.isnan(t_fail): ax.axvline(t_fail,color='red',ls=':',lw=0.8,alpha=0.7)

# 下行左：高振幅速率爆发诊断
ax_r=fig1.add_subplot(gs1[2,:2])
tc=(t_bins_r[:-1]+t_bins_r[1:])/2
ax_r.fill_between(tc,hit_rate_ab,alpha=0.35,color='steelblue',label=f'>{AMP_HIGH_THRESH:.0f}dB 速率')
ax_r.plot(tc,hit_rate_ab_sm,color='navy',lw=1.2,label='平滑速率')
ax_r.axhline(burst_thresh,color='red',lw=1.5,ls='--',label=f'阈值 {burst_thresh:.4f}')
for bt in burst_bin_times: ax_r.axvspan(bt,bt+RATE_BIN,alpha=0.2,color='orange',zorder=0)
ax_r.set_yscale('log'); ax_r.set_xlabel('时间 (s)'); ax_r.set_ylabel('速率 (hits/s)')
ax_r.set_title(f'高振幅速率  橙色=爆发段  ~{t_burst_start:.0f}s起')
ax_r.legend(fontsize=8); vfail([ax_r],t_fail)

# 下行右：全段振幅分布堆叠
ax_d=fig1.add_subplot(gs1[2,2:])
ae2=np.arange(35,105,2); ac2=(ae2[:-1]+ae2[1:])/2
h_ct,_=np.histogram(ae_amp[ contam],bins=ae2)
h_cl,_=np.histogram(ae_amp[~contam],bins=ae2)
ax_d.bar(ac2,h_ct,width=1.8,color='tomato',   alpha=0.8,label=f'条带内(删除) ({n_contam})')
ax_d.bar(ac2,h_cl,width=1.8,color='steelblue',alpha=0.7,bottom=h_ct,label=f'条带外(保留) ({n_clean})')
ax_d.set_xlabel('振幅 (dB)'); ax_d.set_ylabel('频数')
ax_d.set_title('全段振幅分布（堆叠）'); ax_d.legend(fontsize=8)

plt.tight_layout()
out1=os.path.join(RESULT_DIR,'v14_01_条带跟踪诊断.png')
fig1.savefig(out1); plt.close(fig1)
print(f"图1已保存: {out1}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图2  全段干扰对比（6通道），叠加干扰中心曲线
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("生成全段对比图（6通道）...")
fig2=plt.figure(figsize=(20,28))
gs2=gridspec.GridSpec(6,2,figure=fig2,hspace=0.10,wspace=0.06,
                      top=0.93,bottom=0.04,left=0.07,right=0.97)
fig2.suptitle(
    f'声发射振幅 全段干扰对比 v14 — 时变条带跟踪 (0-{T_MAX:.0f}s)\n'
    f'干扰 {n_contam} ({pct:.1f}%)  真实AE {n_clean} ({100-pct:.1f}%)',
    fontsize=10,fontweight='bold')

for ch in range(1,7):
    row=ch-1
    d_ct=ae_contam[ae_contam['CH']==ch]; d_cl=ae_clean[ae_clean['CH']==ch]
    hw=CH_STRIPE_HW.get(ch,6)

    axL=fig2.add_subplot(gs2[row,0])
    if len(d_ct): axL.scatter(d_ct['Time'],d_ct['AMP'],s=SZ,alpha=AL*0.5,
                               color='silver',zorder=1,rasterized=True)
    if len(d_cl): axL.scatter(d_cl['Time'],d_cl['AMP'],s=SZ,alpha=AL,
                               color=CH_COLORS[ch],zorder=2,rasterized=True)
    # 叠加干扰中心曲线
    if ch in ch_center_curve:
        tc_arr,bc_arr=ch_center_curve[ch]
        axL.plot(tc_arr,bc_arr,color='red',lw=1.0,alpha=0.85,zorder=5)
        axL.plot(tc_arr,bc_arr+hw,color='red',lw=0.6,ls='--',alpha=0.6,zorder=5)
        axL.plot(tc_arr,bc_arr-hw,color='red',lw=0.6,ls='--',alpha=0.6,zorder=5)
    for bt in burst_bin_times: axL.axvspan(bt,bt+RATE_BIN,alpha=0.06,color='limegreen',zorder=0)
    axL.set_ylabel(f'CH{ch}\n振幅(dB)',fontsize=9)
    axL.set_ylim(YL); axL.set_xlim(0,T_MAX)
    axL.yaxis.set_major_locator(MultipleLocator(20)); axL.yaxis.set_minor_locator(MultipleLocator(10))
    if row==0:
        axL.set_title(f'原始 ({n_hits} hits)  红线=干扰中心曲线±条带',fontsize=11,pad=8)

    axR=fig2.add_subplot(gs2[row,1],sharey=axL)
    if len(d_cl): axR.scatter(d_cl['Time'],d_cl['AMP'],s=SZ,alpha=AL,
                               color=CH_COLORS[ch],rasterized=True)
    for bt in burst_bin_times: axR.axvspan(bt,bt+RATE_BIN,alpha=0.08,color='limegreen',zorder=0)
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

out2=os.path.join(RESULT_DIR,'v14_02_全段干扰对比图.png')
fig2.savefig(out2); plt.close(fig2)
print(f"图2已保存: {out2}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图3  综合分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig3,axes3=plt.subplots(4,1,figsize=(16,18),sharex=True)
fig3.suptitle('超声波与声发射综合分析 v14\nVp(AIC calibrated) + AE(时变条带跟踪去干扰)',
              fontsize=12,fontweight='bold')
ax3a,ax3b,ax3c,ax3d=axes3

mv=~np.isnan(us_Vp_aic)
ax3a.plot(us_ts[mv],us_Vp_aic[mv],color='crimson',lw=1,alpha=0.85,label='Vp AIC')
ax3a.fill_between(us_ts[mv],us_Vp_aic[mv],alpha=0.12,color='crimson')
ax3a.set_ylabel('Vp (km/s)')
ax3a.set_title(f'P波速度  初始{np.nanmean(us_Vp_aic[valid_aic][:10]):.2f}km/s  峰值{np.nanmax(us_Vp_aic):.2f}km/s')
ax3a.yaxis.set_minor_locator(AutoMinorLocator()); ax3a.legend(loc='upper left')

for ch in range(1,7):
    d=ae_clean[ae_clean['CH']==ch]
    ax3b.scatter(d['Time'],d['AMP'],s=1.5,alpha=0.35,color=CH_COLORS[ch],rasterized=True)
for bt in burst_bin_times: ax3b.axvspan(bt,bt+RATE_BIN,alpha=0.05,color='limegreen',zorder=0)
ax3b.set_ylabel('AE振幅 (dB)')
ax3b.set_title(f'声发射振幅 (v14 去干扰后，爆发段~{t_burst_start:.0f}s起)')
ax3b.legend(handles=[plt.Line2D([0],[0],marker='o',ls='None',color=CH_COLORS[i+1],
            markersize=5,label=f'CH{i+1}') for i in range(6)],
            ncol=6,fontsize=8,loc='upper left',framealpha=0.6)

ax3c.scatter(ae_clean['Time'],ae_clean['ABS_E'],s=1.5,alpha=0.3,color='darkred',rasterized=True)
ax3c.set_yscale('log'); ax3c.set_ylabel('绝对能量 (aJ)'); ax3c.set_title('声发射绝对能量')

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
out3=os.path.join(RESULT_DIR,'v14_03_综合分析.png')
fig3.savefig(out3); plt.close(fig3)
print(f"图3已保存: {out3}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图4  Vp 详图
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig4,axes4=plt.subplots(2,2,figsize=(16,10),sharex='col')
fig4.suptitle(f'P波速度演化 (对零AIC校正)  t_cal={sys_delay:.3f} us',fontsize=12,fontweight='bold')
(ax4a,ax4b),(ax4c,ax4d)=axes4
ax4a.plot(us_ts,us_pt_sw, color='navy',  lw=0.8,alpha=0.8,label='软件')
ax4a.plot(us_ts,us_pt_aic,color='crimson',lw=0.8,alpha=0.8,label='AIC')
ax4a.set_ylabel('P波到时(us)'); ax4a.set_title('P波到时'); ax4a.legend()
ax4b.plot(us_ts,us_pt_aic-us_pt_sw,color='purple',lw=0.6,alpha=0.7)
ax4b.axhline(np.nanmean(diff_valid),color='red',lw=1,ls=':',label=f'均值{np.nanmean(diff_valid):.2f}us')
ax4b.set_ylabel('AIC-软件(us)'); ax4b.set_title('拾取差值'); ax4b.legend()
vb=~(np.isnan(us_Vp_aic)|np.isnan(us_Vp_sw))
ax4c.plot(us_ts[vb],us_Vp_sw[vb], color='navy',  lw=0.8,alpha=0.8,label='Vp(软件)')
ax4c.plot(us_ts[vb],us_Vp_aic[vb],color='crimson',lw=0.8,alpha=0.8,label='Vp(AIC)')
ax4c.set_ylabel('Vp(km/s)'); ax4c.set_xlabel('时间(s)'); ax4c.legend()
ax4c.set_title('P波速度'); ax4c.yaxis.set_minor_locator(AutoMinorLocator())
lim=[min(np.nanmin(us_Vp_sw[vb]),np.nanmin(us_Vp_aic[vb]))*0.95,
     max(np.nanmax(us_Vp_sw[vb]),np.nanmax(us_Vp_aic[vb]))*1.05]
ax4d.scatter(us_Vp_sw[vb],us_Vp_aic[vb],s=3,alpha=0.4,color='steelblue')
ax4d.plot(lim,lim,'r--',lw=1,label='1:1')
ax4d.set_xlim(lim); ax4d.set_ylim(lim); ax4d.legend(); ax4d.set_title('散点对比')
vfail([ax4a,ax4c],t_fail)
plt.tight_layout()
out4=os.path.join(RESULT_DIR,'v14_04_P波速度.png')
fig4.savefig(out4); plt.close(fig4)
print(f"图4已保存: {out4}")

out5=None
if len(evts)>0 and 'x' in evts.columns:
    fig5,axes5=plt.subplots(1,3,figsize=(18,7))
    fig5.suptitle(f'声发射事件空间分布 ({len(evts)} 事件)',fontsize=12,fontweight='bold')
    sk=dict(c=evts['time'],cmap='plasma',s=4,alpha=0.5)
    axes5[0].scatter(evts['x'],evts['y'],**sk); axes5[0].set_xlabel('x(mm)'); axes5[0].set_ylabel('y(mm)'); axes5[0].set_title('XY')
    sc=axes5[1].scatter(evts['x'],evts['z'],**sk); axes5[1].set_xlabel('x(mm)'); axes5[1].set_ylabel('z(mm)'); axes5[1].set_title('XZ')
    axes5[2].scatter(evts['y'],evts['z'],**sk); axes5[2].set_xlabel('y(mm)'); axes5[2].set_ylabel('z(mm)'); axes5[2].set_title('YZ')
    plt.colorbar(sc,ax=axes5[2],label='时间(s)',shrink=0.8)
    plt.tight_layout()
    out5=os.path.join(RESULT_DIR,'v14_05_AE事件空间分布.png')
    fig5.savefig(out5); plt.close(fig5)
    print(f"图5已保存: {out5}")

# ─── CSV ─────────────────────────────────────────────────────────────────
out_vp=os.path.join(RESULT_DIR,'v14_Vp_AIC.csv')
pd.DataFrame({'time_s':us_ts,'pwave_sw_us':us_pt_sw,'pwave_aic_us':us_pt_aic,
              'travel_aic_us':us_pt_aic-sys_delay,'Vp_sw_km':us_Vp_sw,
              'Vp_aic_km':us_Vp_aic}).to_csv(out_vp,index=False)
out_ae=os.path.join(RESULT_DIR,'v14_AE_clean.csv');       ae_clean.to_csv(out_ae,  index=False)
out_ct=os.path.join(RESULT_DIR,'v14_AE_contaminated.csv'); ae_contam.to_csv(out_ct, index=False)

# ─── 汇总 ─────────────────────────────────────────────────────────────────
print("\n"+"="*60)
print("[超声波 Vp]")
print(f"  t_cal={sys_delay:.3f}us  初始Vp={np.nanmean(us_Vp_aic[valid_aic][:10]):.3f}km/s  峰值Vp={np.nanmax(us_Vp_aic):.3f}km/s")
if not np.isnan(t_fail):
    pfi=np.where(fail_mask)[0][0]
    print(f"  破坏前5次均值={np.nanmean(us_Vp_aic[max(0,pfi-5):pfi]):.3f}km/s  破坏时刻={t_fail:.1f}s")
print(f"\n[声发射干扰识别 v14 — 时变振幅条带跟踪]")
print(f"  参考窗={US_MASK_POST_REF}s  T_BIN={T_BIN}s  σ={SIGMA_BINS}bins")
for ch in range(1,7):
    hw=CH_STRIPE_HW.get(ch,6)
    cm=ae_ch==ch
    if ch in CH_NO_FILTER:
        print(f"  CH{ch}: 不过滤  保留 {cm.sum()}")
    else:
        print(f"  CH{ch}: ±{hw}dB  删 {int((contam&cm).sum())}  留 {int((~contam&cm).sum())}")
print(f"  总干扰: {n_contam} ({pct:.1f}%)  真实AE: {n_clean} ({100-pct:.1f}%)")
print(f"\n[输出目录] {RESULT_DIR}")
for f in [out1,out2,out3,out4,out_vp,out_ae,out_ct]+([out5] if out5 else []):
    print(f"  {os.path.basename(f)}")
print("\n分析完成！")
