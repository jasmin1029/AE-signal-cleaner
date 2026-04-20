#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花岗岩单轴压缩试验 - 综合分析 v5
核心逻辑变化：
  60 dB 附近的连续条带 = 超声波干扰，全段全部剔除
  判断顺序：
    Step-1  用基础时间窗（50ms/500ms）锁定"确认干扰"样本
    Step-2  从确认干扰样本的振幅分布，精确拟合干扰振幅带 [AMP_LOW, AMP_HIGH]
    Step-3  全段（全程 0-1730 s）剔除落在该振幅带内的所有撞击
    Step-4  时间窗作为补充：去除振幅带以外但仍在窗内的撞击
  最终干扰 = (振幅在干扰带内) | (在基础时间窗内)

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
for _f in ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams['font.family'] = _f
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'axes.grid': True, 'grid.alpha': 0.25,
    'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight'})

# ─── 路径 ─────────────────────────────────────────────────────────────────
BASE     = r'g:\Cursor project\ZCY-shengfashe'
US_FILE  = os.path.join(BASE, '超声波', '04-15 - ultrasonics data.csv')
CAL_FILE = os.path.join(BASE, '超声波', 'chushi.csv')
AE_HITS  = os.path.join(BASE, '声发射', '04-15-hits-振铃计数、能量等.TXT')
AE_EVTS  = os.path.join(BASE, '声发射', '04-15-声发射事件.TXT')
RESULT_DIR = os.path.join(BASE, '结果')
os.makedirs(RESULT_DIR, exist_ok=True)

# ─── 参数 ─────────────────────────────────────────────────────────────────
H_MM  = 100.0;  H_M = H_MM / 1000.0
FS_HZ = 40e6
BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER = 50e3, 700e3, 4
AIC_SEARCH_OFFSET_US = 12.0
AIC_SEARCH_WIDTH_US  = 30.0
AIC_GLOBAL_START_US  = 5.0

# 干扰识别参数
US_MASK_PRE  = 0.05   # 基础时间窗：激发前 50 ms
US_MASK_POST = 0.50   # 基础时间窗：激发后 500 ms
# 振幅带拟合参数
T_EARLY      = 200.0  # 仅用实验前期数据拟合干扰振幅带（加载初期真实AE极少）
BND_FRAC     = 0.30   # 在平滑峰高 30% 处截断 → 接近 FWHM，避免带子过宽

COLORS    = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
CH_COLORS = {i+1: COLORS[i] for i in range(6)}


# ═══════════════════════════════════════════════════════════════════════════
# § 0  辅助函数
# ═══════════════════════════════════════════════════════════════════════════
def butter_bp_sos(fs, lo, hi, order=4):
    nyq = fs / 2
    return butter(order, [lo/nyq, hi/nyq], btype='band', output='sos')

def apply_bp(sig, sos):
    try:    return sosfiltfilt(sos, sig.astype(float))
    except: return sig.astype(float)

def aic_pick(wf, t_us, s0, s1):
    N  = len(wf)
    i0 = max(1, int(np.searchsorted(t_us, s0)))
    i1 = min(N-1, int(np.searchsorted(t_us, s1)))
    if i1 <= i0+2: return np.nan, None, (i0, i1)
    x  = wf - wf.mean()
    cs = np.cumsum(x*x);  tot = cs[-1]+1e-30
    k  = np.arange(i0, i1)
    v1 = np.maximum(cs[k-1]/k, 1e-30)
    v2 = np.maximum((tot-cs[k-1])/(N-k), 1e-30)
    aic_s = np.convolve(k*np.log(v1)+(N-k)*np.log(v2), np.ones(5)/5, mode='same')
    return t_us[k[np.argmin(aic_s)]], aic_s, (i0, i1)


# ═══════════════════════════════════════════════════════════════════════════
# § 1  超声波数据 & AIC 拾取
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("加载超声波测试数据...")
us_raw   = pd.read_csv(US_FILE, header=None, low_memory=False, dtype=str)
us_ts    = pd.to_numeric(us_raw.iloc[2, 1:], errors='coerce').dropna().values
n_sw     = len(us_ts)
us_pt_sw = pd.to_numeric(us_raw.iloc[5, 1:n_sw+1], errors='coerce').values
wf_block   = us_raw.iloc[7:, :]
wf_time_us = pd.to_numeric(wf_block.iloc[:, 0], errors='coerce').values
wf_data    = wf_block.iloc[:, 1:n_sw+1].apply(pd.to_numeric, errors='coerce').values
del us_raw, wf_block; gc.collect()
n_samp = wf_data.shape[0]
print(f"  扫描: {n_sw}  样点: {n_samp}  时间: {us_ts[0]:.1f}-{us_ts[-1]:.1f} s")

sos_bp = butter_bp_sos(FS_HZ, BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER)

# 对零校准
print("\n对零校准...")
t_cal_aic = np.nan
try:
    cal_raw  = pd.read_csv(CAL_FILE, header=None, skiprows=154,
                           low_memory=False, dtype=str, encoding='latin-1')
    cal_time = pd.to_numeric(cal_raw.iloc[:, 0], errors='coerce').values
    cal_sig  = pd.to_numeric(cal_raw.iloc[:, 1], errors='coerce').values
    ok = ~(np.isnan(cal_time)|np.isnan(cal_sig))
    cal_time, cal_sig = cal_time[ok], cal_sig[ok]
    snr = np.max(np.abs(cal_sig))/(np.std(cal_sig[:100])+1e-30)
    print(f"  SNR={snr:.0f}")
    if snr > 10:
        t_cal_aic, _, _ = aic_pick(apply_bp(cal_sig, sos_bp), cal_time, 0.5, 40.0)
        print(f"  t_cal = {t_cal_aic:.3f} us")
except Exception as e:
    print(f"  警告: {e}")
    cal_time = cal_sig = np.array([0.0])

if np.isnan(t_cal_aic):
    early_sw = us_pt_sw[~np.isnan(us_pt_sw)][:20]
    t_cal_aic = float(np.nanmedian(early_sw) - H_M/4800.0*1e6) if len(early_sw) else 0.0
sys_delay = t_cal_aic

print("\nAIC 拾取...")
t_arr = wf_time_us.copy()
us_pt_aic = np.full(n_sw, np.nan)
for i in range(n_sw):
    wf = wf_data[:, i].astype(float)
    if np.sum(np.isnan(wf)) > n_samp*0.5: continue
    wf[np.isnan(wf)] = 0.0
    ref = us_pt_sw[i] if not np.isnan(us_pt_sw[i]) else 30.0
    us_pt_aic[i], _, _ = aic_pick(apply_bp(wf, sos_bp), t_arr,
                                   max(AIC_GLOBAL_START_US, ref-AIC_SEARCH_OFFSET_US),
                                   ref+AIC_SEARCH_WIDTH_US)

valid_aic = ~np.isnan(us_pt_aic)
travel_aic = (us_pt_aic - sys_delay)*1e-6
us_Vp_aic  = np.where((travel_aic>5e-6)&(travel_aic<200e-6), H_M/travel_aic/1000., np.nan)
travel_sw  = (us_pt_sw  - sys_delay)*1e-6
us_Vp_sw   = np.where((travel_sw >5e-6)&(travel_sw <200e-6), H_M/travel_sw /1000., np.nan)
diff_valid = (us_pt_aic-us_pt_sw)[valid_aic&~np.isnan(us_pt_sw)]
print(f"  成功: {valid_aic.sum()}/{n_sw}  Vp: {np.nanmin(us_Vp_aic):.2f}-{np.nanmax(us_Vp_aic):.2f} km/s")

vp_base  = np.nanmedian(us_Vp_aic[valid_aic][:30]) if valid_aic.sum()>=30 else np.nan
fail_mask= (~np.isnan(us_Vp_aic))&(us_Vp_aic<vp_base*0.7) if not np.isnan(vp_base) else np.zeros(n_sw,bool)
t_fail   = float(us_ts[np.where(fail_mask)[0][0]]) if np.any(fail_mask) else np.nan


# ═══════════════════════════════════════════════════════════════════════════
# § 2  声发射数据
# ═══════════════════════════════════════════════════════════════════════════
print("\n加载声发射数据...")

def parse_ae_hits(path):
    rows = []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) >= 9:
                try:
                    rows.append({'Time':float(p[1]),'CH':int(p[2]),
                                 'RISE':int(p[3]),'COUN':int(p[4]),
                                 'ENER':int(p[5]),'DURATION':int(p[6]),
                                 'AMP':float(p[7]),'ABS_E':float(p[8])})
                except: pass
    return pd.DataFrame(rows)

ae     = parse_ae_hits(AE_HITS)
ae     = ae[ae['Time']>0].sort_values('Time').reset_index(drop=True)
ae_t   = ae['Time'].values
ae_amp = ae['AMP'].values
ae_ch  = ae['CH'].values
n_hits = len(ae)
print(f"  {n_hits} hits  {ae_t.min():.1f}-{ae_t.max():.1f} s")

def parse_ae_events(path):
    events, cur = [], None
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('* Gp#'):
                try:
                    x = float(line.split('x,y,z =')[1].split(',')[0])
                    y = float(line.split('x,y,z =')[1].split(',')[1])
                    z = float(line.split('x,y,z =')[1].split(',')[2].split(',')[0])
                    cur = {'x':x,'y':y,'z':z,'time':None}; events.append(cur)
                except: cur = None
            elif line.startswith('*') and cur is not None:
                p = line.lstrip('*').split()
                if len(p)>=8:
                    try:
                        t = float(p[0])
                        if cur['time'] is None: cur['time'] = t
                    except: pass
    return pd.DataFrame([e for e in events if e['time'] is not None])

evts = parse_ae_events(AE_EVTS)
print(f"  已定位事件: {len(evts)}")


# ═══════════════════════════════════════════════════════════════════════════
# § 3  干扰识别
#
#  Step-1  基础时间窗 → 锁定"确认干扰"样本
#  Step-2  从确认干扰的振幅分布 → 精确确定干扰振幅带
#  Step-3  全段剔除振幅带内所有撞击
#  Step-4  时间窗补充：去除带外仍在窗内的撞击
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("干扰识别 (v5: 振幅带全段剔除 + 时间窗补充)...")

period_approx = float(np.median(np.diff(us_ts)))

# ── Step-1：基础时间窗锁定确认干扰 ────────────────────────────────────────
us_ts_sorted = np.sort(us_ts)
_idx   = np.searchsorted(us_ts_sorted, ae_t, side='right') - 1
dt_from_us = np.where(_idx>=0, ae_t - us_ts_sorted[np.maximum(_idx,0)], np.nan)
_idx_nxt   = np.searchsorted(us_ts_sorted, ae_t, side='left')
dt_to_next = np.where(_idx_nxt<len(us_ts),
                       us_ts_sorted[np.minimum(_idx_nxt, len(us_ts)-1)] - ae_t, np.inf)

contam_tw = ((dt_from_us >= -US_MASK_PRE) & (dt_from_us <= US_MASK_POST)) | \
            ((dt_to_next  >=  0)           & (dt_to_next  <= US_MASK_PRE))

n_tw = int(contam_tw.sum())
print(f"  Step-1 时间窗确认干扰: {n_tw} hits ({100.*n_tw/n_hits:.1f}%)")

# ── Step-2：从确认干扰推断振幅带 ─────────────────────────────────────────
# 只用实验前期 + 时间窗内的撞击（此时真实AE极少，干扰信号最纯净）
early_tw_mask = contam_tw & (ae_t < T_EARLY)
amp_confirmed = ae_amp[early_tw_mask]

if len(amp_confirmed) < 30:
    # 若早期样本不足，退回到全段时间窗内的撞击
    amp_confirmed = ae_amp[contam_tw]
    print(f"  (早期样本不足，使用全段时间窗内撞击 n={len(amp_confirmed)})")
else:
    print(f"  (使用前 {T_EARLY:.0f}s 时间窗内撞击 n={len(amp_confirmed)} 拟合干扰振幅带)")

# 1 dB 分辨率直方图
amp_edges_fine = np.arange(35, 106, 1)
amp_ctrs_fine  = (amp_edges_fine[:-1] + amp_edges_fine[1:]) / 2
hist_conf, _   = np.histogram(amp_confirmed, bins=amp_edges_fine)

# 高斯平滑后找主峰（干扰峰）
hist_smooth = gaussian_filter1d(hist_conf.astype(float), sigma=2)
peak_idx    = int(np.argmax(hist_smooth))
amp_peak    = float(amp_ctrs_fine[peak_idx])

# 从主峰向两侧扩展至峰高的 BND_FRAC 处截断（约 FWHM 级别）
peak_val  = hist_smooth[peak_idx]

left_idx = peak_idx
while left_idx > 0 and hist_smooth[left_idx] > peak_val * BND_FRAC:
    left_idx -= 1
right_idx = peak_idx
while right_idx < len(hist_smooth)-1 and hist_smooth[right_idx] > peak_val * BND_FRAC:
    right_idx += 1

AMP_BAND_LOW  = float(amp_ctrs_fine[max(left_idx  - 1, 0)])
AMP_BAND_HIGH = float(amp_ctrs_fine[min(right_idx + 1, len(amp_ctrs_fine)-1)])

print(f"  Step-2 振幅带拟合结果:")
print(f"    干扰峰值振幅: {amp_peak:.1f} dB")
print(f"    振幅带范围:   [{AMP_BAND_LOW:.1f}, {AMP_BAND_HIGH:.1f}] dB")

# ── Step-3：全段剔除振幅带内所有撞击 ────────────────────────────────────
in_amp_band = (ae_amp >= AMP_BAND_LOW) & (ae_amp <= AMP_BAND_HIGH)
n_band      = int(in_amp_band.sum())
print(f"  Step-3 振幅带全段剔除: {n_band} hits ({100.*n_band/n_hits:.1f}%)")

# ── Step-4：时间窗补充（去除带外仍在窗内的撞击）─────────────────────────
contam_tw_extra = contam_tw & ~in_amp_band
n_tw_extra      = int(contam_tw_extra.sum())
print(f"  Step-4 时间窗补充剔除: {n_tw_extra} hits ({100.*n_tw_extra/n_hits:.1f}%)")

# ── 合并 ─────────────────────────────────────────────────────────────────
contam   = in_amp_band | contam_tw
n_contam = int(contam.sum())
n_clean  = n_hits - n_contam
pct      = 100.0 * n_contam / n_hits

ae_clean  = ae[~contam].reset_index(drop=True)
ae_contam = ae[contam].reset_index(drop=True)

print(f"\n  原始: {n_hits}  干扰: {n_contam} ({pct:.1f}%)  真实AE: {n_clean} ({100-pct:.1f}%)")
print(f"  振幅带外的真实AE振幅范围: "
      f"[{ae_amp[~contam].min():.0f}, {ae_amp[~contam].max():.0f}] dB")


# ═══════════════════════════════════════════════════════════════════════════
# § 4  绘图
# ═══════════════════════════════════════════════════════════════════════════
print("\n生成图表...")

def vline_fail(axes_list, tf):
    if not np.isnan(tf):
        for ax in axes_list:
            ax.axvline(tf, color='red', ls='--', lw=1.2, alpha=0.7)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图1  振幅带拟合诊断
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle(f'干扰振幅带确定  峰值={amp_peak:.1f} dB  带=[{AMP_BAND_LOW:.1f}, {AMP_BAND_HIGH:.1f}] dB',
              fontsize=12, fontweight='bold')

# 确认干扰的振幅分布（用于拟合）
ax = axes1[0]
ax.bar(amp_ctrs_fine, hist_conf, width=0.9, color='tomato', alpha=0.6, label='确认干扰(时间窗内)')
ax.plot(amp_ctrs_fine, hist_smooth, color='darkred', lw=2, label='高斯平滑')
ax.axvspan(AMP_BAND_LOW, AMP_BAND_HIGH, alpha=0.2, color='red', label=f'干扰带 [{AMP_BAND_LOW:.1f},{AMP_BAND_HIGH:.1f}]dB')
ax.axvline(AMP_BAND_LOW,  color='red', lw=1.5, ls='--')
ax.axvline(AMP_BAND_HIGH, color='red', lw=1.5, ls='--')
ax.axvline(amp_peak,      color='darkred', lw=2, ls='-', label=f'峰值 {amp_peak:.1f} dB')
ax.set_xlabel('振幅 (dB)'); ax.set_ylabel('频数')
ax.set_title(f'时间窗内确认干扰振幅分布\n(n={n_tw}, 用于拟合振幅带)')
ax.legend(fontsize=8)

# 全段振幅分布（所有撞击）
ax = axes1[1]
amp_e2  = np.arange(35, 105, 2)
amp_c2  = (amp_e2[:-1]+amp_e2[1:])/2
h_all,  _ = np.histogram(ae_amp,         bins=amp_e2)
h_band, _ = np.histogram(ae_amp[in_amp_band], bins=amp_e2)
h_tw_x, _ = np.histogram(ae_amp[contam_tw_extra], bins=amp_e2)
h_cl,   _ = np.histogram(ae_amp[~contam],bins=amp_e2)
ax.bar(amp_c2, h_band, width=1.8, color='tomato',   alpha=0.8, label=f'振幅带剔除 ({n_band})')
ax.bar(amp_c2, h_tw_x, width=1.8, color='orange',   alpha=0.7,
       bottom=h_band, label=f'时间窗补充剔除 ({n_tw_extra})')
ax.bar(amp_c2, h_cl,   width=1.8, color='steelblue', alpha=0.7,
       bottom=h_band+h_tw_x, label=f'真实AE ({n_clean})')
ax.axvspan(AMP_BAND_LOW, AMP_BAND_HIGH, alpha=0.12, color='red')
ax.axvline(AMP_BAND_LOW,  color='red', lw=1.2, ls='--')
ax.axvline(AMP_BAND_HIGH, color='red', lw=1.2, ls='--')
ax.set_xlabel('振幅 (dB)'); ax.set_ylabel('频数')
ax.set_title('全段振幅分布（堆叠）\n红=振幅带 橙=时间窗补充 蓝=真实AE')
ax.legend(fontsize=8)

# 按通道查看干扰带内 vs 带外比例
ax = axes1[2]
for ch in range(1, 7):
    ch_m  = ae_ch == ch
    n_ch  = ch_m.sum()
    n_b   = (in_amp_band & ch_m).sum()
    n_tw2 = (contam_tw_extra & ch_m).sum()
    n_cl  = (~contam & ch_m).sum()
    ax.bar(ch-0.3, 100.*n_b  /n_ch, 0.25, color='tomato',    label='振幅带' if ch==1 else '')
    ax.bar(ch-0.05,100.*n_tw2/n_ch, 0.25, color='orange',    label='时间窗补' if ch==1 else '')
    ax.bar(ch+0.2, 100.*n_cl /n_ch, 0.25, color='steelblue', label='真实AE'  if ch==1 else '')
ax.set_xlabel('通道'); ax.set_ylabel('占该通道总hits比例 (%)')
ax.set_title('各通道干扰组成')
ax.set_xticks(range(1,7)); ax.legend(fontsize=8)

plt.tight_layout()
out1 = os.path.join(RESULT_DIR, 'v5_01_振幅带拟合诊断.png')
fig1.savefig(out1); plt.close(fig1)
print(f"图1已保存: {out1}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图2  全段干扰对比（6通道 × 全时间）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("生成全段干扰对比图（6通道）...")
T_MAX = ae_t.max() + 20
SZ, AL = 1.5, 0.40
Y_LIM  = [ae_amp.min()-3, ae_amp.max()+3]

fig2 = plt.figure(figsize=(20, 28))
gs2  = gridspec.GridSpec(6, 2, figure=fig2,
                         hspace=0.10, wspace=0.06,
                         top=0.93, bottom=0.04, left=0.07, right=0.97)
fig2.suptitle(
    f'声发射振幅 全段干扰对比 (v5)\n'
    f'振幅带 [{AMP_BAND_LOW:.1f},{AMP_BAND_HIGH:.1f}] dB 全段剔除 + 时间窗补充\n'
    f'共去除 {n_contam} hits ({pct:.1f}%)  保留真实AE {n_clean} hits ({100-pct:.1f}%)',
    fontsize=11, fontweight='bold')

from matplotlib.lines import Line2D

for ch in range(1, 7):
    row  = ch - 1
    d_all = ae[ae_ch == ch]
    d_ct  = ae_contam[ae_contam['CH'] == ch]
    d_cl  = ae_clean[ae_clean['CH']   == ch]

    # 左列：原始（干扰灰色 + 真实彩色）
    axL = fig2.add_subplot(gs2[row, 0])
    if len(d_ct):
        axL.scatter(d_ct['Time'], d_ct['AMP'], s=SZ, alpha=AL*0.5,
                    color='silver', zorder=1, rasterized=True)
    if len(d_cl):
        axL.scatter(d_cl['Time'], d_cl['AMP'], s=SZ, alpha=AL,
                    color=CH_COLORS[ch], zorder=2, rasterized=True)
    # 干扰振幅带参考线
    axL.axhspan(AMP_BAND_LOW, AMP_BAND_HIGH, alpha=0.10, color='red', zorder=0)
    axL.axhline(AMP_BAND_LOW,  color='red', lw=0.8, ls='--', alpha=0.7)
    axL.axhline(AMP_BAND_HIGH, color='red', lw=0.8, ls='--', alpha=0.7)
    axL.text(T_MAX*0.01, (AMP_BAND_LOW+AMP_BAND_HIGH)/2,
             f'干扰带\n{AMP_BAND_LOW:.0f}-{AMP_BAND_HIGH:.0f}dB',
             fontsize=6, color='red', va='center', alpha=0.8)
    axL.set_ylabel(f'CH{ch}\n振幅(dB)', fontsize=9)
    axL.set_ylim(Y_LIM); axL.set_xlim(0, T_MAX)
    axL.yaxis.set_major_locator(MultipleLocator(20))
    axL.yaxis.set_minor_locator(MultipleLocator(10))
    if row == 0:
        axL.set_title(f'原始 ({n_hits} hits)', fontsize=11, pad=8)
        axL.legend(handles=[
            Line2D([0],[0],marker='o',ls='None',color='silver',  markersize=4,label=f'干扰 ({n_contam})'),
            Line2D([0],[0],marker='o',ls='None',color=CH_COLORS[ch],markersize=4,label=f'真实AE ({n_clean})')],
            fontsize=8, loc='upper left')

    # 右列：去干扰后
    axR = fig2.add_subplot(gs2[row, 1], sharey=axL)
    if len(d_cl):
        axR.scatter(d_cl['Time'], d_cl['AMP'], s=SZ, alpha=AL,
                    color=CH_COLORS[ch], rasterized=True)
    axR.axhspan(AMP_BAND_LOW, AMP_BAND_HIGH, alpha=0.05, color='red', zorder=0)
    axR.axhline(AMP_BAND_LOW,  color='red', lw=0.6, ls='--', alpha=0.4)
    axR.axhline(AMP_BAND_HIGH, color='red', lw=0.6, ls='--', alpha=0.4)
    axR.set_xlim(0, T_MAX); axR.set_ylim(Y_LIM)
    axR.yaxis.set_major_locator(MultipleLocator(20))
    axR.tick_params(labelleft=False)
    if row == 0:
        axR.set_title(f'去干扰后 ({n_clean} hits)', fontsize=11, pad=8)
    if row < 5:
        axL.tick_params(labelbottom=False); axR.tick_params(labelbottom=False)
    else:
        axL.set_xlabel('时间 (s)'); axR.set_xlabel('时间 (s)')
    if not np.isnan(t_fail):
        axL.axvline(t_fail, color='red', lw=0.8, ls=':', alpha=0.6)
        axR.axvline(t_fail, color='red', lw=0.8, ls=':', alpha=0.6)

out2 = os.path.join(RESULT_DIR, 'v5_02_全段干扰对比图.png')
fig2.savefig(out2); plt.close(fig2)
print(f"图2已保存: {out2}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图3  Vp 演化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10), sharex='col')
fig3.suptitle(f'P波速度演化 (对零AIC校正)  t_cal={sys_delay:.3f} us', fontsize=12, fontweight='bold')
(ax3a, ax3b), (ax3c, ax3d) = axes3
ax3a.plot(us_ts, us_pt_sw,  color='navy',    lw=0.8, alpha=0.8, label='软件')
ax3a.plot(us_ts, us_pt_aic, color='crimson', lw=0.8, alpha=0.8, label='AIC')
ax3a.set_ylabel('P波到时 (us)'); ax3a.set_title('P波到时'); ax3a.legend()
ax3b.plot(us_ts, us_pt_aic-us_pt_sw, color='purple', lw=0.6, alpha=0.7)
ax3b.axhline(np.nanmean(diff_valid),color='red',lw=1,ls=':',label=f'均值{np.nanmean(diff_valid):.2f}us')
ax3b.set_ylabel('AIC-软件 (us)'); ax3b.set_title('拾取差值'); ax3b.legend()
vb = ~(np.isnan(us_Vp_aic)|np.isnan(us_Vp_sw))
ax3c.plot(us_ts[vb], us_Vp_sw[vb],  color='navy',    lw=0.8, alpha=0.8, label='Vp(软件)')
ax3c.plot(us_ts[vb], us_Vp_aic[vb], color='crimson', lw=0.8, alpha=0.8, label='Vp(AIC)')
ax3c.set_ylabel('Vp (km/s)'); ax3c.set_xlabel('时间 (s)')
ax3c.set_title(f'P波速度  Vp={H_MM:.0f}mm/(t_AIC-{sys_delay:.2f}us)')
ax3c.legend(); ax3c.yaxis.set_minor_locator(AutoMinorLocator())
lim = [min(np.nanmin(us_Vp_sw[vb]),np.nanmin(us_Vp_aic[vb]))*0.95,
       max(np.nanmax(us_Vp_sw[vb]),np.nanmax(us_Vp_aic[vb]))*1.05]
ax3d.scatter(us_Vp_sw[vb], us_Vp_aic[vb], s=3, alpha=0.4, color='steelblue')
ax3d.plot(lim, lim, 'r--', lw=1, label='1:1')
ax3d.set_xlabel('Vp软件'); ax3d.set_ylabel('Vp AIC'); ax3d.legend()
ax3d.set_xlim(lim); ax3d.set_ylim(lim); ax3d.set_title('散点对比')
vline_fail([ax3a, ax3c], t_fail)
plt.tight_layout()
out3 = os.path.join(RESULT_DIR, 'v5_03_P波速度.png')
fig3.savefig(out3); plt.close(fig3)
print(f"图3已保存: {out3}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图4  综合分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig4, axes4 = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
fig4.suptitle('超声波与声发射综合分析 v5\n'
              'Vp (AIC, calibrated) + AE (amplitude-band + time-window removal)',
              fontsize=12, fontweight='bold')
ax4a, ax4b, ax4c, ax4d = axes4

m_vp = ~np.isnan(us_Vp_aic)
ax4a.plot(us_ts[m_vp], us_Vp_aic[m_vp], color='crimson', lw=1, alpha=0.85, label='Vp AIC')
ax4a.fill_between(us_ts[m_vp], us_Vp_aic[m_vp], alpha=0.12, color='crimson')
ax4a.set_ylabel('Vp (km/s)')
ax4a.set_title(f'P波速度演化  初始{np.nanmean(us_Vp_aic[valid_aic][:10]):.2f} km/s  峰值{np.nanmax(us_Vp_aic):.2f} km/s')
ax4a.yaxis.set_minor_locator(AutoMinorLocator()); ax4a.legend(loc='upper left')

for ch in range(1, 7):
    d = ae_clean[ae_clean['CH'] == ch]
    ax4b.scatter(d['Time'], d['AMP'], s=1.5, alpha=0.35,
                 color=CH_COLORS[ch], label=f'CH{ch}', rasterized=True)
ax4b.axhspan(AMP_BAND_LOW, AMP_BAND_HIGH, alpha=0.06, color='red')
ax4b.axhline(AMP_BAND_LOW,  color='red', lw=0.8, ls=':', alpha=0.5, label=f'已剔除带边界')
ax4b.axhline(AMP_BAND_HIGH, color='red', lw=0.8, ls=':', alpha=0.5)
ax4b.set_ylabel('AE振幅 (dB)')
ax4b.set_title(f'声发射振幅 (v5: 干扰带[{AMP_BAND_LOW:.0f},{AMP_BAND_HIGH:.0f}]dB 全段剔除后)')
ax4b.legend(handles=[plt.Line2D([0],[0],marker='o',ls='None',
            color=CH_COLORS[i+1],markersize=5,label=f'CH{i+1}') for i in range(6)],
            ncol=6, fontsize=8, loc='upper left', framealpha=0.6)

ax4c.scatter(ae_clean['Time'], ae_clean['ABS_E'],
             s=1.5, alpha=0.3, color='darkred', rasterized=True)
ax4c.set_yscale('log')
ax4c.set_ylabel('绝对能量 (aJ)')
ax4c.set_title('声发射绝对能量 (去干扰后)')

ae_cl_s = ae_clean.sort_values('Time')
ax4d.plot(ae['Time'], np.arange(1, n_hits+1), color='tomato',
          lw=1.2, ls='--', alpha=0.7, label=f'原始 ({n_hits})')
ax4d.plot(ae_cl_s['Time'], np.arange(1, n_clean+1), color='steelblue',
          lw=1.5, alpha=0.9, label=f'去干扰后 ({n_clean})')
ax4d.set_ylabel('累计撞击数'); ax4d.set_xlabel('时间 (s)')
ax4d.set_title('累计撞击数对比'); ax4d.legend()

vline_fail(axes4, t_fail)
if not np.isnan(t_fail):
    axes4[0].text(t_fail+15, ax4a.get_ylim()[0]*1.02, f'破坏 {t_fail:.0f}s', color='red', fontsize=9)

plt.tight_layout()
out4 = os.path.join(RESULT_DIR, 'v5_04_综合分析.png')
fig4.savefig(out4); plt.close(fig4)
print(f"图4已保存: {out4}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图5  AE 事件空间分布
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
out5 = None
if len(evts) > 0 and 'x' in evts.columns:
    fig5, axes5 = plt.subplots(1, 3, figsize=(18, 7))
    fig5.suptitle(f'声发射事件空间分布 ({len(evts)} 事件)', fontsize=12, fontweight='bold')
    skw = dict(c=evts['time'], cmap='plasma', s=4, alpha=0.5)
    axes5[0].scatter(evts['x'], evts['y'], **skw); axes5[0].set_xlabel('x(mm)'); axes5[0].set_ylabel('y(mm)'); axes5[0].set_title('XY')
    sc = axes5[1].scatter(evts['x'], evts['z'], **skw); axes5[1].set_xlabel('x(mm)'); axes5[1].set_ylabel('z(mm)'); axes5[1].set_title('XZ')
    axes5[2].scatter(evts['y'], evts['z'], **skw); axes5[2].set_xlabel('y(mm)'); axes5[2].set_ylabel('z(mm)'); axes5[2].set_title('YZ')
    plt.colorbar(sc, ax=axes5[2], label='时间 (s)', shrink=0.8)
    plt.tight_layout()
    out5 = os.path.join(RESULT_DIR, 'v5_05_AE事件空间分布.png')
    fig5.savefig(out5); plt.close(fig5)
    print(f"图5已保存: {out5}")

# ─── CSV 输出 ─────────────────────────────────────────────────────────────
out_vp = os.path.join(RESULT_DIR, 'v5_Vp_AIC.csv')
pd.DataFrame({'time_s':us_ts,'pwave_sw_us':us_pt_sw,'pwave_aic_us':us_pt_aic,
              'travel_aic_us':us_pt_aic-sys_delay,'Vp_sw_km':us_Vp_sw,
              'Vp_aic_km':us_Vp_aic}).to_csv(out_vp, index=False)

out_ae = os.path.join(RESULT_DIR, 'v5_AE_clean.csv')
ae_clean.to_csv(out_ae, index=False)

out_ct = os.path.join(RESULT_DIR, 'v5_AE_contaminated.csv')
ae_contam.to_csv(out_ct, index=False)

# ─── 统计汇总 ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("统计汇总")
print("="*60)
print(f"\n[超声波 Vp]")
print(f"  系统延时 t_cal:  {sys_delay:.3f} us")
print(f"  初始 Vp:         {np.nanmean(us_Vp_aic[valid_aic][:10]):.3f} km/s")
print(f"  峰值 Vp:         {np.nanmax(us_Vp_aic):.3f} km/s")
if not np.isnan(t_fail):
    pfi = np.where(fail_mask)[0][0]
    print(f"  破坏前5次均值:   {np.nanmean(us_Vp_aic[max(0,pfi-5):pfi]):.3f} km/s")
    print(f"  推测破坏时刻:    {t_fail:.1f} s")

print(f"\n[声发射干扰识别 v5]")
print(f"  干扰振幅带:      [{AMP_BAND_LOW:.1f}, {AMP_BAND_HIGH:.1f}] dB  (峰值 {amp_peak:.1f} dB)")
print(f"  振幅带全段剔除:  {n_band} hits ({100.*n_band/n_hits:.1f}%)")
print(f"  时间窗补充剔除:  {n_tw_extra} hits ({100.*n_tw_extra/n_hits:.1f}%)")
print(f"  总干扰:          {n_contam} hits ({pct:.1f}%)")
print(f"  真实AE:          {n_clean} hits ({100-pct:.1f}%)")
print(f"  保留AE振幅范围:  [{ae_amp[~contam].min():.0f}, {ae_amp[~contam].max():.0f}] dB")

print(f"\n[输出目录] {RESULT_DIR}")
for f in [out1,out2,out3,out4,out_vp,out_ae,out_ct]+([out5] if out5 else []):
    print(f"  {os.path.basename(f)}")
print("\n分析完成！")
