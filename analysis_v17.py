#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花岗岩单轴压缩试验 - 综合分析 v17
干扰识别：时变振幅条带（纯振幅，无时间窗条件）
  核心改进：
    ① 安静期：条带内所有点直接删除（不再依赖时间窗），清除全部聚集干扰
    ② 爆发段（近破坏）：条带内随机保留 50%（稀释，而非全删或全保）
    ③ 拟合仍用安静期时间窗参考hits（不受影响）
    ④ 外推为恒值（安静期拟合段末端均值）
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

# ── 时间窗（安静期过滤条件①，延长至1000ms捕捉持续干扰）─────────────────────
US_MASK_PRE  = 0.05   # US 脉冲前 50ms
US_MASK_POST = 1.00   # US 脉冲后 1000ms（v16: 从500ms延长至1000ms）

# ── 条带半宽（过滤条件②，单位 dB）──────────────────────────────────────────
# 爆发段完全不过滤，故此参数仅影响安静期
CH_STRIPE_HW = {
    1: 5,    # ±5 dB
    2: 5,
    3: 4,
    4: 0,    # CH4 不过滤
    5: 6,
    6: 5,
}

CH_NO_FILTER = {4}

# ── 条带拟合参数 ─────────────────────────────────────────────────────────
T_BIN        = 60.0   # 时间箱宽度（s）
SIGMA_BINS   = 2      # 高斯平滑宽度（bins）
MIN_REF_N    = 3      # 每箱最少参考 hits
FIT_END_MARGIN = 200.0  # 安静期末端去掉多少秒不参与拟合（避免爆发前AE污染）
EXTRAP_LAST_N  = 5      # 恒值外推：取安静期最后多少个有效箱均值

# ── 爆发段检测参数 ───────────────────────────────────────────────────────
AMP_HIGH_THRESH = 74.0
RATE_BIN    = 5.0
RATE_SMOOTH = 3
T_EARLY     = 200.0

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
# § 3  干扰识别 v16
# ═══════════════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("干扰识别 v17（安静期：纯条带全删  爆发段：条带内50%稀释）...")

us_ts_s = np.sort(us_ts)
period_approx = float(np.median(np.diff(us_ts_s)))

# ── 时间窗掩膜（条件①） ─────────────────────────────────────────────────
_idx      = np.searchsorted(us_ts_s, ae_t, side='right') - 1
dt_after  = np.where(_idx>=0, ae_t - us_ts_s[np.maximum(_idx,0)], np.nan)
_idx_n    = np.searchsorted(us_ts_s, ae_t, side='left')
dt_before = np.where(_idx_n<len(us_ts_s), us_ts_s[np.minimum(_idx_n,len(us_ts_s)-1)] - ae_t, np.inf)
in_tw = ((dt_after>=0)&(dt_after<=US_MASK_POST)) | \
        ((dt_before>=0)&(dt_before<=US_MASK_PRE))

# ── 先做爆发段检测 ───────────────────────────────────────────────────────
above_mask = ae_amp > AMP_HIGH_THRESH
t_bins_r   = np.arange(0, ae_t.max()+RATE_BIN, RATE_BIN)
hit_cnt_ab,_= np.histogram(ae_t[above_mask], bins=t_bins_r)
hit_rate_ab = (hit_cnt_ab/RATE_BIN).astype(float)
kernel      = np.ones(RATE_SMOOTH)/RATE_SMOOTH
hit_rate_sm = np.convolve(hit_rate_ab, kernel, mode='same')
early_b     = t_bins_r[:-1] < T_EARLY
baseline_ab = max(float(np.median(hit_rate_sm[early_b])),0.005) if early_b.any() else 0.01
burst_thresh= max(baseline_ab*3.0, 0.05)
is_burst_bin= hit_rate_sm > burst_thresh
bin_idx     = np.clip(np.searchsorted(t_bins_r[1:],ae_t),0,len(hit_cnt_ab)-1)
is_burst    = is_burst_bin[bin_idx]
t_burst_start = float(t_bins_r[np.where(is_burst_bin)[0][0]]) if is_burst_bin.any() else ae_t.max()
burst_bin_times = t_bins_r[:-1][is_burst_bin]
print(f"  US周期={period_approx:.2f}s  爆发段起始={t_burst_start:.0f}s")
print(f"  爆发段（{t_burst_start:.0f}s起）条带内50%稀释，共 {is_burst.sum()} hits")

# ── 逐通道拟合干扰条带中心（仅用安静期头部参考hits，末端留出FIT_END_MARGIN） ──
ae_fine = np.arange(35,106,1); ac_fine=(ae_fine[:-1]+ae_fine[1:])/2
t_bins_all   = np.arange(0, ae_t.max()+T_BIN, T_BIN)
t_centers_all= (t_bins_all[:-1]+t_bins_all[1:])/2

ch_center_curve = {}
interf_center   = np.full(n_hits, np.nan)

print(f"\n  {'CH':>3}  {'半宽':>6}  {'拟合参考':>8}  {'有效箱':>6}  {'拟合期中心范围':>18}  恒值外推")
for ch in range(1,7):
    hw = CH_STRIPE_HW.get(ch, 5)
    if ch in CH_NO_FILTER or hw == 0:
        print(f"  CH{ch}  {'---':>6}  {'---':>8}  {'---':>6}  不过滤")
        continue

    ch_mask  = ae_ch == ch
    # 参考：安静期（爆发前）+ 时间窗内
    ref_quiet= in_tw & ch_mask & ~is_burst
    t_ref    = ae_t[ref_quiet]
    amp_ref  = ae_amp[ref_quiet]

    # 拟合仅用安静期去掉末端FIT_END_MARGIN秒（避免爆发前AE污染峰值）
    t_fit_end = max(t_burst_start - FIT_END_MARGIN, T_BIN * 4)
    fit_sel   = t_ref < t_fit_end
    t_ref_fit = t_ref[fit_sel]
    amp_ref_fit = amp_ref[fit_sel]

    # 安静期时间箱（用于过滤的完整安静期范围）
    t_end_quiet = min(t_burst_start, ae_t.max())
    t_bins_q    = np.arange(0, t_end_quiet+T_BIN, T_BIN)
    t_cq        = (t_bins_q[:-1]+t_bins_q[1:])/2
    n_bq        = len(t_cq)

    bc_q = np.full(n_bq, np.nan)
    for i in range(n_bq):
        sel = (t_ref_fit>=t_bins_q[i]) & (t_ref_fit<t_bins_q[i+1])
        if sel.sum() >= MIN_REF_N:
            h,_ = np.histogram(amp_ref_fit[sel], bins=ae_fine)
            hs  = gaussian_filter1d(h.astype(float), sigma=1)
            bc_q[i] = ac_fine[int(np.argmax(hs))]

    valid_q = ~np.isnan(bc_q)
    n_valid = valid_q.sum()
    if n_valid < 2:
        print(f"  CH{ch}  {hw:>6}  {fit_sel.sum():>8}  {n_valid:>6}  安静参考不足，跳过")
        continue

    # 插值补全安静期箱
    bc_q_filled = np.interp(t_cq, t_cq[valid_q], bc_q[valid_q])
    bc_q_smooth = gaussian_filter1d(bc_q_filled, sigma=SIGMA_BINS)

    # 恒值外推：取安静期最后 EXTRAP_LAST_N 个有效箱（落在拟合范围内）的均值
    fit_bin_mask = t_cq < t_fit_end
    last_valid_infit = bc_q_smooth[fit_bin_mask]
    if len(last_valid_infit) >= EXTRAP_LAST_N:
        extrap_val = float(np.mean(last_valid_infit[-EXTRAP_LAST_N:]))
    elif len(last_valid_infit) > 0:
        extrap_val = float(np.mean(last_valid_infit))
    else:
        extrap_val = float(bc_q_smooth[-1])
    last_t = float(t_cq[-1])

    # 合并：安静期插值平滑 + 爆发期恒值
    center_full = np.where(
        t_centers_all <= last_t,
        np.interp(t_centers_all, t_cq, bc_q_smooth),
        extrap_val
    )

    ch_center_curve[ch] = (t_centers_all.copy(), center_full.copy())
    interf_center[ch_mask] = np.interp(ae_t[ch_mask], t_centers_all, center_full)

    cmin,cmax = bc_q_smooth[fit_bin_mask].min() if fit_bin_mask.any() else bc_q_smooth.min(), \
                bc_q_smooth[fit_bin_mask].max() if fit_bin_mask.any() else bc_q_smooth.max()
    print(f"  CH{ch}  {hw:>6}  {fit_sel.sum():>8}  {n_valid:>6}  [{cmin:.1f},{cmax:.1f}] dB"
          f"  → {extrap_val:.1f} dB (恒值)")

# ── 干扰判定：安静期条带内全删；爆发段条带内50%稀释 ──────────────────────
no_filter  = np.isin(ae_ch, list(CH_NO_FILTER))
on_stripe  = np.zeros(n_hits, dtype=bool)
for ch in range(1,7):
    hw = CH_STRIPE_HW.get(ch, 5)
    if ch in CH_NO_FILTER or hw==0: continue
    cm = ae_ch==ch
    valid_c = ~np.isnan(interf_center)
    on_stripe |= cm & valid_c & (np.abs(ae_amp - interf_center) <= hw)

# 安静期：条带内全删（无时间窗条件）
contam_quiet = on_stripe & ~no_filter & ~is_burst

# 爆发段：条带内随机删除50%（稀释，固定种子保证可复现）
rng = np.random.default_rng(42)
rand_mask = rng.random(n_hits) < 0.50   # True = 这条hit被随机选中删除
contam_burst = on_stripe & ~no_filter & is_burst & rand_mask

contam    = contam_quiet | contam_burst
ae_clean  = ae[~contam].reset_index(drop=True)
ae_contam = ae[ contam].reset_index(drop=True)
n_contam  = int(contam.sum()); n_clean=n_hits-n_contam; pct=100.*n_contam/n_hits

print(f"\n  [统计]  安静期条带内（全删）: {int(contam_quiet.sum())}  "
      f"爆发段条带内（50%稀释删）: {int(contam_burst.sum())}  "
      f"爆发段条带内（50%保留）: {int((on_stripe&~no_filter&is_burst&~rand_mask).sum())}")
for ch in range(1,7):
    cm=ae_ch==ch
    note=' ← 不过滤' if ch in CH_NO_FILTER else ''
    burst_del = int((contam_burst&cm).sum())
    burst_keep= int(((on_stripe|~on_stripe)&is_burst&~no_filter&cm&~contam_burst).sum())
    print(f"    CH{ch}: 删 {int((contam&cm).sum()):>5}  留 {int((~contam&cm).sum()):>5}"
          f"  (爆发段删 {burst_del} 留 {int((is_burst&cm&~contam).sum())}){note}")
print(f"  总干扰: {n_contam} ({pct:.1f}%)  真实AE: {n_clean} ({100-pct:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# § 4  绘图
# ═══════════════════════════════════════════════════════════════════════════
print("\n生成图表...")

def vfail(axes,tf):
    if not np.isnan(tf):
        for ax in axes: ax.axvline(tf,color='red',ls='--',lw=1.2,alpha=0.7)

T_MAX=ae_t.max()+20; SZ,AL=1.5,0.40; YL=[ae_amp.min()-3,ae_amp.max()+3]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图1  条带拟合诊断
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig1=plt.figure(figsize=(20,14))
gs1=gridspec.GridSpec(3,4,figure=fig1,hspace=0.45,wspace=0.35,top=0.93,bottom=0.06)
fig1.suptitle(
    f'v17 诊断  纯条带过滤（安静全删 | 爆发50%稀释）\n'
    f'T_BIN={T_BIN}s  σ={SIGMA_BINS}  '
    f'干扰 {n_contam} ({pct:.1f}%)  真实AE {n_clean} ({100-pct:.1f}%)',
    fontsize=10,fontweight='bold')

for ch in range(1,7):
    row_idx,col_idx=(0,ch-1) if ch<=4 else (1,ch-5)
    ax=fig1.add_subplot(gs1[row_idx,col_idx])
    hw=CH_STRIPE_HW.get(ch,5); cm=ae_ch==ch

    # 安静期参考 hits（窗内非爆发）
    ref_q_ch = in_tw & cm & ~is_burst
    other_ch  = ~ref_q_ch & cm
    ax.scatter(ae_t[other_ch],  ae_amp[other_ch],  s=0.8, alpha=0.15, color='gray',   zorder=1, rasterized=True)
    ax.scatter(ae_t[ref_q_ch],  ae_amp[ref_q_ch],  s=1.0, alpha=0.6,  color='tomato', zorder=2, rasterized=True, label=f'安静参考 n={ref_q_ch.sum()}')

    if ch in ch_center_curve:
        tc_arr,bc_arr=ch_center_curve[ch]
        ax.plot(tc_arr,bc_arr,       color='navy',lw=1.5,zorder=4,label='干扰中心')
        ax.plot(tc_arr,bc_arr+hw,    color='navy',lw=0.8,ls='--',alpha=0.7,zorder=4)
        ax.plot(tc_arr,bc_arr-hw,    color='navy',lw=0.8,ls='--',alpha=0.7,zorder=4)
        ax.fill_between(tc_arr,bc_arr-hw,bc_arr+hw,alpha=0.12,color='navy',zorder=3)
        if not np.isnan(t_burst_start):
            ax.axvline(t_burst_start,color='orange',lw=1.2,ls='-',alpha=0.8,label=f'爆发 {t_burst_start:.0f}s')
        # 拟合截止线
        t_fit_end_ch = max(t_burst_start - FIT_END_MARGIN, T_BIN * 4)
        ax.axvline(t_fit_end_ch, color='green', lw=0.8, ls=':', alpha=0.7, label=f'拟合截止 {t_fit_end_ch:.0f}s')

    ax.set_xlim(0,T_MAX); ax.set_ylim(YL)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('振幅 (dB)')
    note='(不过滤)' if ch in CH_NO_FILTER else f'±{hw}dB  删{int((contam&cm).sum())} 留{int((~contam&cm).sum())} (爆~50%)'
    ax.set_title(f'CH{ch}  {note}')
    ax.legend(fontsize=6,loc='upper left')
    if not np.isnan(t_fail): ax.axvline(t_fail,color='red',ls=':',lw=0.8,alpha=0.7)

# 下行左：爆发速率
ax_r=fig1.add_subplot(gs1[2,:2])
tc=(t_bins_r[:-1]+t_bins_r[1:])/2
ax_r.fill_between(tc,hit_rate_ab,alpha=0.35,color='steelblue',label=f'>{AMP_HIGH_THRESH:.0f}dB 速率')
ax_r.plot(tc,hit_rate_sm,color='navy',lw=1.2,label='平滑速率')
ax_r.axhline(burst_thresh,color='red',lw=1.5,ls='--',label=f'阈值 {burst_thresh:.4f}')
for bt in burst_bin_times: ax_r.axvspan(bt,bt+RATE_BIN,alpha=0.2,color='orange',zorder=0)
ax_r.set_yscale('log'); ax_r.set_xlabel('时间 (s)'); ax_r.set_ylabel('速率 (hits/s)')
ax_r.set_title(f'高振幅速率  爆发段~{t_burst_start:.0f}s起（橙色 = 不过滤区域）')
ax_r.legend(fontsize=8); vfail([ax_r],t_fail)

# 下行右：全段振幅分布堆叠
ax_d=fig1.add_subplot(gs1[2,2:])
ae2=np.arange(35,105,2); ac2=(ae2[:-1]+ae2[1:])/2
h_ct,_=np.histogram(ae_amp[ contam],bins=ae2)
h_cl,_=np.histogram(ae_amp[~contam],bins=ae2)
ax_d.bar(ac2,h_ct,width=1.8,color='tomato',   alpha=0.8,label=f'干扰(删除) ({n_contam})')
ax_d.bar(ac2,h_cl,width=1.8,color='steelblue',alpha=0.7,bottom=h_ct,label=f'真实AE(保留) ({n_clean})')
ax_d.set_xlabel('振幅 (dB)'); ax_d.set_ylabel('频数')
ax_d.set_title('全段振幅分布（堆叠）'); ax_d.legend(fontsize=8)

plt.tight_layout()
out1=os.path.join(RESULT_DIR,'v17_01_条带拟合诊断.png')
fig1.savefig(out1); plt.close(fig1)
print(f"图1已保存: {out1}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图2  全段干扰对比（6通道），叠加中心曲线
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("生成全段对比图（6通道）...")
fig2=plt.figure(figsize=(20,28))
gs2=gridspec.GridSpec(6,2,figure=fig2,hspace=0.10,wspace=0.06,
                      top=0.93,bottom=0.04,left=0.07,right=0.97)
fig2.suptitle(
    f'声发射振幅 全段干扰对比 v17 (0-{T_MAX:.0f}s)\n'
    f'纯条带过滤  安静全删 | 爆发50%稀释  干扰 {n_contam} ({pct:.1f}%)  真实AE {n_clean} ({100-pct:.1f}%)',
    fontsize=10,fontweight='bold')

for ch in range(1,7):
    row=ch-1; hw=CH_STRIPE_HW.get(ch,5)
    d_ct=ae_contam[ae_contam['CH']==ch]; d_cl=ae_clean[ae_clean['CH']==ch]

    axL=fig2.add_subplot(gs2[row,0])
    if len(d_ct): axL.scatter(d_ct['Time'],d_ct['AMP'],s=SZ,alpha=AL*0.5,color='silver',zorder=1,rasterized=True)
    if len(d_cl): axL.scatter(d_cl['Time'],d_cl['AMP'],s=SZ,alpha=AL,color=CH_COLORS[ch],zorder=2,rasterized=True)
    if ch in ch_center_curve:
        tc_arr,bc_arr=ch_center_curve[ch]
        axL.plot(tc_arr,bc_arr,    color='red',lw=0.9,alpha=0.9,zorder=5)
        axL.plot(tc_arr,bc_arr+hw, color='red',lw=0.6,ls='--',alpha=0.6,zorder=5)
        axL.plot(tc_arr,bc_arr-hw, color='red',lw=0.6,ls='--',alpha=0.6,zorder=5)
    # 爆发段橙色底（这些区域完全不过滤）
    for bt in burst_bin_times: axL.axvspan(bt,bt+RATE_BIN,alpha=0.08,color='orange',zorder=0)
    axL.set_ylabel(f'CH{ch}\n振幅(dB)',fontsize=9)
    axL.set_ylim(YL); axL.set_xlim(0,T_MAX)
    axL.yaxis.set_major_locator(MultipleLocator(20)); axL.yaxis.set_minor_locator(MultipleLocator(10))
    if row==0: axL.set_title(f'原始 ({n_hits} hits)  红线=干扰中心±条带  橙色=爆发(50%稀释)',fontsize=11,pad=8)

    axR=fig2.add_subplot(gs2[row,1],sharey=axL)
    if len(d_cl): axR.scatter(d_cl['Time'],d_cl['AMP'],s=SZ,alpha=AL,color=CH_COLORS[ch],rasterized=True)
    for bt in burst_bin_times: axR.axvspan(bt,bt+RATE_BIN,alpha=0.10,color='orange',zorder=0)
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

out2=os.path.join(RESULT_DIR,'v17_02_全段干扰对比图.png')
fig2.savefig(out2); plt.close(fig2)
print(f"图2已保存: {out2}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图3  综合分析 / 图4  Vp / 图5  空间分布
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig3,axes3=plt.subplots(4,1,figsize=(16,18),sharex=True)
fig3.suptitle('超声波与声发射综合分析 v17\nVp(AIC) + AE(纯条带：安静全删，爆发50%稀释)',fontsize=12,fontweight='bold')
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
for bt in burst_bin_times: ax3b.axvspan(bt,bt+RATE_BIN,alpha=0.05,color='orange',zorder=0)
ax3b.set_ylabel('AE振幅 (dB)'); ax3b.set_title(f'声发射振幅 (v17 去干扰后，爆发段50%稀释)')
ax3b.legend(handles=[plt.Line2D([0],[0],marker='o',ls='None',color=CH_COLORS[i+1],
            markersize=5,label=f'CH{i+1}') for i in range(6)],ncol=6,fontsize=8,loc='upper left')
ax3c.scatter(ae_clean['Time'],ae_clean['ABS_E'],s=1.5,alpha=0.3,color='darkred',rasterized=True)
ax3c.set_yscale('log'); ax3c.set_ylabel('绝对能量 (aJ)'); ax3c.set_title('声发射绝对能量')
ax3d.plot(ae['Time'],np.arange(1,n_hits+1),color='tomato',lw=1.2,ls='--',alpha=0.7,label=f'原始 ({n_hits})')
ax3d.plot(ae_clean.sort_values('Time')['Time'],np.arange(1,n_clean+1),
          color='steelblue',lw=1.5,alpha=0.9,label=f'去干扰后 ({n_clean})')
ax3d.set_ylabel('累计撞击数'); ax3d.set_xlabel('时间 (s)'); ax3d.set_title('累计撞击数对比'); ax3d.legend()
vfail(axes3,t_fail)
plt.tight_layout()
out3=os.path.join(RESULT_DIR,'v17_03_综合分析.png'); fig3.savefig(out3); plt.close(fig3)
print(f"图3已保存: {out3}")

fig4,axes4=plt.subplots(2,2,figsize=(16,10),sharex='col')
fig4.suptitle(f'P波速度演化  t_cal={sys_delay:.3f} us',fontsize=12,fontweight='bold')
(ax4a,ax4b),(ax4c,ax4d)=axes4
ax4a.plot(us_ts,us_pt_sw,color='navy',lw=0.8,alpha=0.8,label='软件')
ax4a.plot(us_ts,us_pt_aic,color='crimson',lw=0.8,alpha=0.8,label='AIC')
ax4a.set_ylabel('P波到时(us)'); ax4a.set_title('P波到时'); ax4a.legend()
ax4b.plot(us_ts,us_pt_aic-us_pt_sw,color='purple',lw=0.6,alpha=0.7)
ax4b.axhline(np.nanmean(diff_valid),color='red',lw=1,ls=':',label=f'均值{np.nanmean(diff_valid):.2f}us')
ax4b.set_ylabel('AIC-软件(us)'); ax4b.set_title('拾取差值'); ax4b.legend()
vb=~(np.isnan(us_Vp_aic)|np.isnan(us_Vp_sw))
ax4c.plot(us_ts[vb],us_Vp_sw[vb],color='navy',lw=0.8,alpha=0.8,label='Vp(软件)')
ax4c.plot(us_ts[vb],us_Vp_aic[vb],color='crimson',lw=0.8,alpha=0.8,label='Vp(AIC)')
ax4c.set_ylabel('Vp(km/s)'); ax4c.set_xlabel('时间(s)'); ax4c.legend()
ax4c.set_title('P波速度'); ax4c.yaxis.set_minor_locator(AutoMinorLocator())
lim=[min(np.nanmin(us_Vp_sw[vb]),np.nanmin(us_Vp_aic[vb]))*0.95,
     max(np.nanmax(us_Vp_sw[vb]),np.nanmax(us_Vp_aic[vb]))*1.05]
ax4d.scatter(us_Vp_sw[vb],us_Vp_aic[vb],s=3,alpha=0.4,color='steelblue')
ax4d.plot(lim,lim,'r--',lw=1,label='1:1'); ax4d.set_xlim(lim); ax4d.set_ylim(lim)
ax4d.legend(); ax4d.set_title('散点对比')
vfail([ax4a,ax4c],t_fail)
plt.tight_layout()
out4=os.path.join(RESULT_DIR,'v17_04_P波速度.png'); fig4.savefig(out4); plt.close(fig4)
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
    out5=os.path.join(RESULT_DIR,'v17_05_AE事件空间分布.png')
    fig5.savefig(out5); plt.close(fig5)
    print(f"图5已保存: {out5}")

# ─── CSV ─────────────────────────────────────────────────────────────────
out_vp=os.path.join(RESULT_DIR,'v17_Vp_AIC.csv')
pd.DataFrame({'time_s':us_ts,'pwave_sw_us':us_pt_sw,'pwave_aic_us':us_pt_aic,
              'travel_aic_us':us_pt_aic-sys_delay,'Vp_sw_km':us_Vp_sw,
              'Vp_aic_km':us_Vp_aic}).to_csv(out_vp,index=False)
out_ae=os.path.join(RESULT_DIR,'v17_AE_clean.csv');       ae_clean.to_csv(out_ae,  index=False)
out_ct=os.path.join(RESULT_DIR,'v17_AE_contaminated.csv'); ae_contam.to_csv(out_ct, index=False)

print("\n"+"="*60)
print("[超声波 Vp]")
print(f"  t_cal={sys_delay:.3f}us  初始Vp={np.nanmean(us_Vp_aic[valid_aic][:10]):.3f}km/s  峰值Vp={np.nanmax(us_Vp_aic):.3f}km/s")
print(f"\n[声发射干扰识别 v17 — 纯条带（安静全删 | 爆发50%稀释）]")
print(f"  T_BIN={T_BIN}s  σ={SIGMA_BINS}  爆发起={t_burst_start:.0f}s")
print(f"  拟合截止={t_burst_start - FIT_END_MARGIN:.0f}s（安静期末端{FIT_END_MARGIN:.0f}s不参与拟合）")
for ch in range(1,7):
    hw=CH_STRIPE_HW.get(ch,5); cm=ae_ch==ch
    if ch in CH_NO_FILTER: print(f"  CH{ch}: 不过滤  保留 {cm.sum()}")
    else:
        burst_del = int((contam_burst&cm).sum())
        burst_on  = int((on_stripe&~no_filter&is_burst&cm).sum())
        print(f"  CH{ch}: ±{hw}dB  删 {int((contam&cm).sum())}  留 {int((~contam&cm).sum())}"
              f"  (爆发段条带内 {burst_on} 删~50% → 删{burst_del})")
print(f"  总干扰: {n_contam} ({pct:.1f}%)  真实AE: {n_clean} ({100-pct:.1f}%)")
print(f"\n[输出目录] {RESULT_DIR}")
for f in [out1,out2,out3,out4,out_vp,out_ae,out_ct]+([out5] if out5 else []):
    print(f"  {os.path.basename(f)}")
print("\n分析完成！")
