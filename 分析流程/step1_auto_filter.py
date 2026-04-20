#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_auto_filter.py — 第一步：自动干扰滤波
================================================
流程:
  1. 超声波数据 AIC 拾取 → P 波速度 Vp(t)
  2. 声发射 hits 加载
  3. 爆发段检测（高振幅速率法）
  4. 逐通道拟合干扰条带中心（时间变化的振幅带）
  5. 干扰判定：
       安静期：条带内全部删除
       爆发段：条带内随机保留 BURST_DILUTE 比例（默认 50%）
  6. 输出 CSV + 诊断图

输出目录: 结果/step1_自动滤波/
"""

import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, gc, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# ── 导入配置 ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

os.makedirs(STEP1_DIR, exist_ok=True)

# ── 中文字体 ───────────────────────────────────────────────────────────────
for _f in ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams['font.family'] = _f; break
    except Exception:
        pass
plt.rcParams.update({
    'axes.unicode_minus': False,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'axes.grid': True,
    'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': SAVE_DPI, 'savefig.bbox': 'tight',
})


# ═══════════════════════════════════════════════════════════════════════════
# § 0  工具函数
# ═══════════════════════════════════════════════════════════════════════════
def bp_sos(fs, lo, hi, order=4):
    nyq = fs / 2
    return butter(order, [lo/nyq, hi/nyq], btype='band', output='sos')

def apply_bp(sig, sos):
    try:
        return sosfiltfilt(sos, sig.astype(float))
    except Exception:
        return sig.astype(float)

def aic_pick(wf, t, s0, s1):
    N  = len(wf)
    i0 = max(1, int(np.searchsorted(t, s0)))
    i1 = min(N-1, int(np.searchsorted(t, s1)))
    if i1 <= i0 + 2:
        return np.nan
    x  = wf - wf.mean()
    cs = np.cumsum(x * x); tot = cs[-1] + 1e-30
    k  = np.arange(i0, i1)
    v1 = np.maximum(cs[k-1] / k,       1e-30)
    v2 = np.maximum((tot - cs[k-1]) / (N - k), 1e-30)
    aic = np.convolve(k*np.log(v1) + (N-k)*np.log(v2), np.ones(5)/5, mode='same')
    return t[k[np.argmin(aic)]]

def parse_ae_hits(path):
    rows = []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) >= 9:
                try:
                    rows.append({'Time': float(p[1]), 'CH': int(p[2]),
                                 'RISE': int(p[3]),   'COUN': int(p[4]),
                                 'ENER': int(p[5]),   'DURATION': int(p[6]),
                                 'AMP': float(p[7]),  'ABS_E': float(p[8])})
                except (ValueError, IndexError):
                    pass
    df = pd.DataFrame(rows)
    return df[df['Time'] > 0].sort_values('Time').reset_index(drop=True) if len(df) else df


# ═══════════════════════════════════════════════════════════════════════════
# § 1  超声波 AIC 拾取 + Vp
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1 — 自动干扰滤波")
print("=" * 60)
print("\n[1/5] 超声波 AIC 拾取...")

us_raw     = pd.read_csv(US_FILE, header=None, low_memory=False, dtype=str)
us_ts      = pd.to_numeric(us_raw.iloc[2, 1:], errors='coerce').dropna().values
n_sw       = len(us_ts)
us_pt_sw   = pd.to_numeric(us_raw.iloc[5, 1:n_sw+1], errors='coerce').values
wf_block   = us_raw.iloc[7:, :]
wf_time_us = pd.to_numeric(wf_block.iloc[:, 0], errors='coerce').values
wf_data    = wf_block.iloc[:, 1:n_sw+1].apply(pd.to_numeric, errors='coerce').values
del us_raw, wf_block; gc.collect()
n_samp = wf_data.shape[0]
print(f"  {n_sw} sweeps  {us_ts[0]:.1f}–{us_ts[-1]:.1f} s")

sos_bp = bp_sos(FS_HZ, BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER)

# 零校准
t_cal = np.nan
try:
    cal = pd.read_csv(CAL_FILE, header=None, skiprows=154,
                      low_memory=False, dtype=str, encoding='latin-1')
    ct  = pd.to_numeric(cal.iloc[:, 0], errors='coerce').values
    cs_ = pd.to_numeric(cal.iloc[:, 1], errors='coerce').values
    ok  = ~(np.isnan(ct) | np.isnan(cs_)); ct, cs_ = ct[ok], cs_[ok]
    snr = np.max(np.abs(cs_)) / (np.std(cs_[:100]) + 1e-30)
    if snr > 10:
        t_cal = aic_pick(apply_bp(cs_, sos_bp), ct, 0.5, 40.0)
    print(f"  SNR={snr:.0f}  t_cal={t_cal:.3f} μs")
except Exception as e:
    print(f"  零校准警告: {e}")

if np.isnan(t_cal):
    es = us_pt_sw[~np.isnan(us_pt_sw)][:20]
    t_cal = float(np.nanmedian(es) - H_M / 4800. * 1e6) if len(es) else 0.0
sys_delay = t_cal

# AIC 逐 sweep 拾取
us_pt_aic = np.full(n_sw, np.nan)
for i in range(n_sw):
    wf = wf_data[:, i].astype(float)
    if np.sum(np.isnan(wf)) > n_samp * 0.5:
        continue
    wf[np.isnan(wf)] = 0.0
    ref = us_pt_sw[i] if not np.isnan(us_pt_sw[i]) else 30.0
    us_pt_aic[i] = aic_pick(apply_bp(wf, sos_bp), wf_time_us,
                             max(AIC_START_US, ref - AIC_OFFSET_US),
                             ref + AIC_WIDTH_US)

valid_aic = ~np.isnan(us_pt_aic)
tr_aic = (us_pt_aic - sys_delay) * 1e-6
tr_sw  = (us_pt_sw  - sys_delay) * 1e-6
us_Vp_aic = np.where((tr_aic > 5e-6) & (tr_aic < 200e-6), H_M / tr_aic / 1000., np.nan)
us_Vp_sw  = np.where((tr_sw  > 5e-6) & (tr_sw  < 200e-6), H_M / tr_sw  / 1000., np.nan)
diff_valid = (us_pt_aic - us_pt_sw)[valid_aic & ~np.isnan(us_pt_sw)]
print(f"  Vp(AIC): {np.nanmin(us_Vp_aic):.2f}–{np.nanmax(us_Vp_aic):.2f} km/s")

vp_base   = np.nanmedian(us_Vp_aic[valid_aic][:30]) if valid_aic.sum() >= 30 else np.nan
fail_mask = (~np.isnan(us_Vp_aic)) & (us_Vp_aic < vp_base * 0.7) if not np.isnan(vp_base) \
            else np.zeros(n_sw, bool)
t_fail    = float(us_ts[np.where(fail_mask)[0][0]]) if np.any(fail_mask) else np.nan
print(f"  推测破坏时刻: {t_fail:.1f} s" if not np.isnan(t_fail) else "  未检测到明显破坏时刻")


# ═══════════════════════════════════════════════════════════════════════════
# § 2  声发射数据加载
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/5] 加载声发射数据...")
ae     = parse_ae_hits(AE_HITS)
ae_t   = ae['Time'].values
ae_amp = ae['AMP'].values
ae_ch  = ae['CH'].values
n_hits = len(ae)
print(f"  {n_hits} hits  CH={sorted(ae['CH'].unique())}  "
      f"时间 {ae_t.min():.1f}–{ae_t.max():.1f} s")

# 声发射事件（用于空间分布图）
def parse_ae_events(path):
    events, cur = [], None
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('* Gp#'):
                try:
                    parts = line.split('x,y,z =')[1].split(',')
                    cur = {'x': float(parts[0]), 'y': float(parts[1]),
                           'z': float(parts[2].split(',')[0]), 'time': None}
                    events.append(cur)
                except Exception:
                    cur = None
            elif line.startswith('*') and cur is not None:
                p = line.lstrip('*').split()
                if len(p) >= 8:
                    try:
                        if cur['time'] is None:
                            cur['time'] = float(p[0])
                    except Exception:
                        pass
    return pd.DataFrame([e for e in events if e['time'] is not None])

evts = parse_ae_events(AE_EVTS)
print(f"  {len(evts)} 已定位事件")


# ═══════════════════════════════════════════════════════════════════════════
# § 3  爆发段检测
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/5] 爆发段检测...")
us_ts_s       = np.sort(us_ts)
period_approx = float(np.median(np.diff(us_ts_s)))

above_mask  = ae_amp > AMP_HIGH_THRESH
t_bins_r    = np.arange(0, ae_t.max() + RATE_BIN, RATE_BIN)
cnt_ab, _   = np.histogram(ae_t[above_mask], bins=t_bins_r)
rate_ab     = (cnt_ab / RATE_BIN).astype(float)
kernel      = np.ones(RATE_SMOOTH) / RATE_SMOOTH
rate_sm     = np.convolve(rate_ab, kernel, mode='same')
early_b     = t_bins_r[:-1] < T_EARLY
baseline    = max(float(np.median(rate_sm[early_b])), 0.005) if early_b.any() else 0.01
burst_thr   = max(baseline * BURST_RATE_MULT, 0.05)
is_burst_bin= rate_sm > burst_thr
bin_idx     = np.clip(np.searchsorted(t_bins_r[1:], ae_t), 0, len(cnt_ab)-1)
is_burst    = is_burst_bin[bin_idx]
t_burst_start = float(t_bins_r[np.where(is_burst_bin)[0][0]]) \
                if is_burst_bin.any() else ae_t.max()
burst_bins  = t_bins_r[:-1][is_burst_bin]
print(f"  US 周期={period_approx:.2f} s  爆发段起始={t_burst_start:.0f} s  "
      f"爆发段 hits={is_burst.sum()}")

# 时间窗掩膜（用于拟合参考 hits）
_idx      = np.searchsorted(us_ts_s, ae_t, side='right') - 1
dt_after  = np.where(_idx >= 0, ae_t - us_ts_s[np.maximum(_idx, 0)], np.nan)
_idx_n    = np.searchsorted(us_ts_s, ae_t, side='left')
dt_before = np.where(_idx_n < len(us_ts_s),
                     us_ts_s[np.minimum(_idx_n, len(us_ts_s)-1)] - ae_t, np.inf)
in_tw = ((dt_after  >= 0) & (dt_after  <= US_MASK_POST)) | \
        ((dt_before >= 0) & (dt_before <= US_MASK_PRE))


# ═══════════════════════════════════════════════════════════════════════════
# § 4  逐通道拟合干扰条带中心
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4/5] 拟合干扰条带中心...")
ae_fine  = np.arange(35, 106, 1); ac_fine = (ae_fine[:-1] + ae_fine[1:]) / 2
t_bins_all    = np.arange(0, ae_t.max() + T_BIN, T_BIN)
t_centers_all = (t_bins_all[:-1] + t_bins_all[1:]) / 2

ch_center_curve = {}          # ch -> (t_arr, center_arr)
interf_center   = np.full(n_hits, np.nan)

print(f"  {'CH':>3}  {'半宽':>5}  {'参考hits':>8}  {'有效箱':>6}  {'中心范围 (dB)':>18}  恒值外推")
for ch in range(1, 7):
    hw = CH_STRIPE_HW.get(ch, 5)
    if ch in CH_NO_FILTER or hw == 0:
        print(f"  CH{ch}  {'—':>5}  {'—':>8}  {'—':>6}  不过滤")
        continue

    ch_mask   = ae_ch == ch
    # 参考：安静期 + 时间窗内，且排除末端 FIT_END_MARGIN 秒
    ref_quiet = in_tw & ch_mask & ~is_burst
    t_ref     = ae_t[ref_quiet]
    amp_ref   = ae_amp[ref_quiet]
    t_fit_end = max(t_burst_start - FIT_END_MARGIN, T_BIN * 4)
    fit_sel   = t_ref < t_fit_end
    t_rf, a_rf = t_ref[fit_sel], amp_ref[fit_sel]

    # 安静期时间箱
    t_bins_q = np.arange(0, min(t_burst_start, ae_t.max()) + T_BIN, T_BIN)
    t_cq     = (t_bins_q[:-1] + t_bins_q[1:]) / 2
    bc_q     = np.full(len(t_cq), np.nan)
    for i in range(len(t_cq)):
        sel = (t_rf >= t_bins_q[i]) & (t_rf < t_bins_q[i+1])
        if sel.sum() >= MIN_REF_N:
            h, _ = np.histogram(a_rf[sel], bins=ae_fine)
            hs   = gaussian_filter1d(h.astype(float), sigma=1)
            bc_q[i] = ac_fine[int(np.argmax(hs))]

    valid_q = ~np.isnan(bc_q)
    if valid_q.sum() < 2:
        print(f"  CH{ch}  {hw:>5}  {fit_sel.sum():>8}  {valid_q.sum():>6}  参考不足，跳过")
        continue

    bc_filled = np.interp(t_cq, t_cq[valid_q], bc_q[valid_q])
    bc_smooth = gaussian_filter1d(bc_filled, sigma=SIGMA_BINS)

    # 恒值外推（取安静期拟合范围末端均值）
    fit_mask = t_cq < t_fit_end
    last_vals = bc_smooth[fit_mask][-EXTRAP_LAST_N:] if fit_mask.any() else bc_smooth[-EXTRAP_LAST_N:]
    extrap_val = float(np.mean(last_vals)) if len(last_vals) else float(bc_smooth[-1])
    last_t     = float(t_cq[-1])

    center_full = np.where(t_centers_all <= last_t,
                           np.interp(t_centers_all, t_cq, bc_smooth),
                           extrap_val)
    ch_center_curve[ch] = (t_centers_all.copy(), center_full.copy())
    interf_center[ch_mask] = np.interp(ae_t[ch_mask], t_centers_all, center_full)

    c_range = bc_smooth[fit_mask] if fit_mask.any() else bc_smooth
    print(f"  CH{ch}  {hw:>5}  {fit_sel.sum():>8}  {valid_q.sum():>6}  "
          f"[{c_range.min():.1f}, {c_range.max():.1f}]  → {extrap_val:.1f} dB")


# ═══════════════════════════════════════════════════════════════════════════
# § 5  干扰判定 + 输出
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5/5] 干扰判定...")
no_filter = np.isin(ae_ch, list(CH_NO_FILTER))
on_stripe = np.zeros(n_hits, dtype=bool)
for ch in range(1, 7):
    hw = CH_STRIPE_HW.get(ch, 5)
    if ch in CH_NO_FILTER or hw == 0:
        continue
    cm      = ae_ch == ch
    valid_c = ~np.isnan(interf_center)
    on_stripe |= cm & valid_c & (np.abs(ae_amp - interf_center) <= hw)

# 安静期：条带内全删
contam_quiet = on_stripe & ~no_filter & ~is_burst

# 爆发段：条带内随机删除 (1 - BURST_DILUTE) 比例
rng       = np.random.default_rng(BURST_RAND_SEED)
rand_mask = rng.random(n_hits) >= BURST_DILUTE   # True = 删除
contam_burst = on_stripe & ~no_filter & is_burst & rand_mask

contam    = contam_quiet | contam_burst
ae_clean  = ae[~contam].reset_index(drop=True)
ae_contam = ae[ contam].reset_index(drop=True)
n_contam  = int(contam.sum()); n_clean = n_hits - n_contam
pct       = 100. * n_contam / n_hits

print(f"\n  {'CH':>3}  {'总计':>6}  {'保留':>6}  {'删除':>6}  {'删除率':>7}")
for ch in range(1, 7):
    cm   = ae_ch == ch
    kept = int((~contam & cm).sum())
    deld = int(( contam & cm).sum())
    flag = '  ← 不过滤' if ch in CH_NO_FILTER else ''
    print(f"  CH{ch}  {cm.sum():>6}  {kept:>6}  {deld:>6}  "
          f"{100.*deld/cm.sum():>6.1f}%{flag}")
print(f"  {'合计':>3}  {n_hits:>6}  {n_clean:>6}  {n_contam:>6}  {pct:>6.1f}%")

# CSV 输出
ae_clean.to_csv(STEP1_AE_CLEAN,  index=False)
ae_contam.to_csv(STEP1_AE_CONTAM, index=False)
vp_df = pd.DataFrame({'time_s': us_ts, 'pwave_sw_us': us_pt_sw,
                       'pwave_aic_us': us_pt_aic,
                       'Vp_sw_km': us_Vp_sw, 'Vp_aic_km': us_Vp_aic})
vp_df.to_csv(os.path.join(STEP1_DIR, 'Vp_AIC.csv'), index=False)
print(f"\n  CSV 已保存至: {STEP1_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# § 6  绘图
# ═══════════════════════════════════════════════════════════════════════════
print("\n生成图表...")
T_MAX = ae_t.max() + 20
YL    = [ae_amp.min() - 3, ae_amp.max() + 3]

def vline_fail(axes_list):
    if not np.isnan(t_fail):
        for ax in axes_list:
            ax.axvline(t_fail, color='red', ls='--', lw=1.0, alpha=0.7)

# ── 图 1：条带拟合诊断 ────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(20, 14))
gs1  = gridspec.GridSpec(3, 4, figure=fig1, hspace=0.48, wspace=0.35,
                          top=0.92, bottom=0.06)
fig1.suptitle(f'条带拟合诊断  干扰 {n_contam} ({pct:.1f}%)  '
              f'保留 {n_clean} ({100-pct:.1f}%)', fontsize=11, fontweight='bold')

for ch in range(1, 7):
    ri, ci = (0, ch-1) if ch <= 4 else (1, ch-5)
    ax = fig1.add_subplot(gs1[ri, ci])
    hw = CH_STRIPE_HW.get(ch, 5); cm = ae_ch == ch
    ref_q  = in_tw & cm & ~is_burst
    other  = ~ref_q & cm
    ax.scatter(ae_t[other],  ae_amp[other],  s=0.8, alpha=0.15, color='gray',   rasterized=True)
    ax.scatter(ae_t[ref_q],  ae_amp[ref_q],  s=1.0, alpha=0.60, color='tomato', rasterized=True,
               label=f'安静参考 n={ref_q.sum()}')
    if ch in ch_center_curve:
        tc, bc = ch_center_curve[ch]
        ax.plot(tc, bc,    color='navy', lw=1.5, label='干扰中心')
        ax.plot(tc, bc+hw, color='navy', lw=0.8, ls='--', alpha=0.7)
        ax.plot(tc, bc-hw, color='navy', lw=0.8, ls='--', alpha=0.7)
        ax.fill_between(tc, bc-hw, bc+hw, alpha=0.12, color='navy')
        ax.axvline(t_burst_start, color='orange', lw=1.0, ls=':', alpha=0.8)
        t_fe = max(t_burst_start - FIT_END_MARGIN, T_BIN*4)
        ax.axvline(t_fe, color='green', lw=0.7, ls=':', alpha=0.6)
    ax.set_xlim(0, T_MAX); ax.set_ylim(YL)
    ax.set_xlabel('时间 (s)'); ax.set_ylabel('振幅 (dB)')
    note = '(不过滤)' if ch in CH_NO_FILTER else f'±{hw} dB  删{int((contam&cm).sum())} 留{int((~contam&cm).sum())}'
    ax.set_title(f'CH{ch}  {note}', fontsize=9)
    ax.legend(fontsize=6, loc='upper left')
    vline_fail([ax])

# 爆发速率
ax_r = fig1.add_subplot(gs1[2, :2])
tc_r = (t_bins_r[:-1] + t_bins_r[1:]) / 2
ax_r.fill_between(tc_r, rate_ab, alpha=0.35, color='steelblue', label=f'>{AMP_HIGH_THRESH:.0f} dB 速率')
ax_r.plot(tc_r, rate_sm, color='navy', lw=1.2, label='平滑速率')
ax_r.axhline(burst_thr, color='red', lw=1.5, ls='--', label=f'阈值 {burst_thr:.3f}')
for bt in burst_bins:
    ax_r.axvspan(bt, bt+RATE_BIN, alpha=0.2, color='orange', zorder=0)
ax_r.set_yscale('log'); ax_r.set_xlabel('时间 (s)'); ax_r.set_ylabel('速率 (hits/s)')
ax_r.set_title(f'高振幅速率  爆发段起={t_burst_start:.0f} s'); ax_r.legend(fontsize=8)
vline_fail([ax_r])

# 振幅分布
ax_d = fig1.add_subplot(gs1[2, 2:])
ae2 = np.arange(35, 105, 2); ac2 = (ae2[:-1] + ae2[1:]) / 2
h_ct, _ = np.histogram(ae_amp[ contam], bins=ae2)
h_cl, _ = np.histogram(ae_amp[~contam], bins=ae2)
ax_d.bar(ac2, h_ct, width=1.8, color='tomato',    alpha=0.8, label=f'干扰 ({n_contam})')
ax_d.bar(ac2, h_cl, width=1.8, color='steelblue', alpha=0.7, bottom=h_ct, label=f'保留 ({n_clean})')
ax_d.set_xlabel('振幅 (dB)'); ax_d.set_ylabel('频数')
ax_d.set_title('振幅分布'); ax_d.legend(fontsize=8)

fig1.savefig(os.path.join(STEP1_DIR, 'fig1_条带拟合诊断.png'))
plt.close(fig1)
print(f"  图1 条带拟合诊断")

# ── 图 2：6 通道干扰对比 ───────────────────────────────────────────────────
fig2 = plt.figure(figsize=(20, 28))
gs2  = gridspec.GridSpec(6, 2, figure=fig2, hspace=0.10, wspace=0.06,
                          top=0.93, bottom=0.04, left=0.07, right=0.97)
fig2.suptitle(f'声发射振幅  自动滤波对比  干扰 {n_contam} ({pct:.1f}%)  保留 {n_clean} ({100-pct:.1f}%)',
              fontsize=11, fontweight='bold')

for ch in range(1, 7):
    row = ch - 1
    hw  = CH_STRIPE_HW.get(ch, 5)
    d_d = ae_contam[ae_contam['CH'] == ch]
    d_k = ae_clean[ae_clean['CH'] == ch]

    axL = fig2.add_subplot(gs2[row, 0])
    if len(d_d):
        axL.scatter(d_d['Time'], d_d['AMP'], s=SCATTER_S, alpha=SCATTER_A*0.5,
                    color='silver', zorder=1, rasterized=True)
    if len(d_k):
        axL.scatter(d_k['Time'], d_k['AMP'], s=SCATTER_S, alpha=SCATTER_A,
                    color=CH_COLORS[ch], zorder=2, rasterized=True)
    if ch in ch_center_curve:
        tc, bc = ch_center_curve[ch]
        axL.plot(tc, bc,    color='red', lw=0.9, alpha=0.9)
        axL.plot(tc, bc+hw, color='red', lw=0.6, ls='--', alpha=0.6)
        axL.plot(tc, bc-hw, color='red', lw=0.6, ls='--', alpha=0.6)
    for bt in burst_bins:
        axL.axvspan(bt, bt+RATE_BIN, alpha=0.07, color='orange', zorder=0)
    axL.set_ylabel(f'CH{ch}\n振幅 (dB)', fontsize=9)
    axL.set_ylim(YL); axL.set_xlim(0, T_MAX)
    axL.yaxis.set_major_locator(MultipleLocator(20))
    axL.yaxis.set_minor_locator(MultipleLocator(10))
    if row == 0:
        axL.set_title(f'原始 ({n_hits} hits)  红线=干扰条带  橙=爆发段', fontsize=10, pad=8)

    axR = fig2.add_subplot(gs2[row, 1], sharey=axL)
    if len(d_k):
        axR.scatter(d_k['Time'], d_k['AMP'], s=SCATTER_S, alpha=SCATTER_A,
                    color=CH_COLORS[ch], rasterized=True)
    for bt in burst_bins:
        axR.axvspan(bt, bt+RATE_BIN, alpha=0.09, color='orange', zorder=0)
    axR.set_xlim(0, T_MAX); axR.set_ylim(YL)
    axR.yaxis.set_major_locator(MultipleLocator(20))
    axR.tick_params(labelleft=False)
    if row == 0:
        axR.set_title(f'自动滤波后 ({n_clean} hits)', fontsize=10, pad=8)
    if row < 5:
        axL.tick_params(labelbottom=False); axR.tick_params(labelbottom=False)
    else:
        axL.set_xlabel('时间 (s)'); axR.set_xlabel('时间 (s)')
    vline_fail([axL, axR])

fig2.savefig(os.path.join(STEP1_DIR, 'fig2_滤波对比.png'))
plt.close(fig2)
print(f"  图2 滤波对比")

# ── 图 3：Vp + AE 综合 ────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
fig3.suptitle('超声波与声发射综合分析\nVp(AIC) + AE 自动滤波后', fontsize=12, fontweight='bold')
ax3a, ax3b, ax3c, ax3d = axes3

mv = ~np.isnan(us_Vp_aic)
ax3a.plot(us_ts[mv], us_Vp_aic[mv], color='crimson', lw=1, alpha=0.85, label='Vp AIC')
ax3a.fill_between(us_ts[mv], us_Vp_aic[mv], alpha=0.12, color='crimson')
ax3a.set_ylabel('Vp (km/s)')
ax3a.set_title(f'P 波速度  初始 {np.nanmean(us_Vp_aic[valid_aic][:10]):.2f} km/s  '
               f'峰值 {np.nanmax(us_Vp_aic):.2f} km/s')
ax3a.yaxis.set_minor_locator(AutoMinorLocator()); ax3a.legend(loc='upper left')

for ch in range(1, 7):
    d = ae_clean[ae_clean['CH'] == ch]
    ax3b.scatter(d['Time'], d['AMP'], s=1.5, alpha=0.35,
                 color=CH_COLORS[ch], rasterized=True)
for bt in burst_bins:
    ax3b.axvspan(bt, bt+RATE_BIN, alpha=0.05, color='orange', zorder=0)
ax3b.set_ylabel('AE 振幅 (dB)'); ax3b.set_title('声发射振幅（自动滤波后）')
ax3b.legend(handles=[plt.Line2D([0],[0], marker='o', ls='None',
            color=CH_COLORS[i+1], markersize=5, label=f'CH{i+1}')
            for i in range(6)], ncol=6, fontsize=8, loc='upper left')

ax3c.scatter(ae_clean['Time'], ae_clean['ABS_E'], s=1.5, alpha=0.3,
             color='darkred', rasterized=True)
ax3c.set_yscale('log'); ax3c.set_ylabel('绝对能量 (aJ)'); ax3c.set_title('声发射绝对能量')

ae_s = ae_clean.sort_values('Time')
ax3d.plot(ae['Time'],  np.arange(1, n_hits+1),  color='tomato',   lw=1.2, ls='--', alpha=0.7, label=f'原始 ({n_hits})')
ax3d.plot(ae_s['Time'], np.arange(1, n_clean+1), color='steelblue', lw=1.5, alpha=0.9, label=f'滤波后 ({n_clean})')
ax3d.set_ylabel('累计 hits'); ax3d.set_xlabel('时间 (s)')
ax3d.set_title('累计 hits 对比'); ax3d.legend()
vline_fail(axes3)

plt.tight_layout()
fig3.savefig(os.path.join(STEP1_DIR, 'fig3_综合分析.png'))
plt.close(fig3)
print(f"  图3 综合分析")

# ── 图 4：Vp 详细 ─────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10), sharex='col')
fig4.suptitle(f'P 波速度演化  t_cal={sys_delay:.3f} μs', fontsize=12, fontweight='bold')
(ax4a, ax4b), (ax4c, ax4d) = axes4
ax4a.plot(us_ts, us_pt_sw,  color='navy',   lw=0.8, alpha=0.8, label='软件')
ax4a.plot(us_ts, us_pt_aic, color='crimson', lw=0.8, alpha=0.8, label='AIC')
ax4a.set_ylabel('P 波到时 (μs)'); ax4a.set_title('P 波到时'); ax4a.legend()
ax4b.plot(us_ts, us_pt_aic - us_pt_sw, color='purple', lw=0.6, alpha=0.7)
ax4b.axhline(np.nanmean(diff_valid), color='red', lw=1, ls=':',
             label=f'均值 {np.nanmean(diff_valid):.2f} μs')
ax4b.set_ylabel('AIC−软件 (μs)'); ax4b.set_title('拾取差值'); ax4b.legend()
vb = ~(np.isnan(us_Vp_aic) | np.isnan(us_Vp_sw))
ax4c.plot(us_ts[vb], us_Vp_sw[vb],  color='navy',   lw=0.8, alpha=0.8, label='Vp 软件')
ax4c.plot(us_ts[vb], us_Vp_aic[vb], color='crimson', lw=0.8, alpha=0.8, label='Vp AIC')
ax4c.set_ylabel('Vp (km/s)'); ax4c.set_xlabel('时间 (s)')
ax4c.set_title('P 波速度'); ax4c.legend(); ax4c.yaxis.set_minor_locator(AutoMinorLocator())
lim = [min(np.nanmin(us_Vp_sw[vb]), np.nanmin(us_Vp_aic[vb])) * 0.95,
       max(np.nanmax(us_Vp_sw[vb]), np.nanmax(us_Vp_aic[vb])) * 1.05]
ax4d.scatter(us_Vp_sw[vb], us_Vp_aic[vb], s=3, alpha=0.4, color='steelblue')
ax4d.plot(lim, lim, 'r--', lw=1, label='1:1')
ax4d.set_xlim(lim); ax4d.set_ylim(lim); ax4d.legend(); ax4d.set_title('散点对比')
vline_fail([ax4a, ax4c])
plt.tight_layout()
fig4.savefig(os.path.join(STEP1_DIR, 'fig4_P波速度.png'))
plt.close(fig4)
print(f"  图4 P 波速度")

# ── 图 5：AE 事件空间分布 ──────────────────────────────────────────────────
if len(evts) > 0 and 'x' in evts.columns:
    fig5, axes5 = plt.subplots(1, 3, figsize=(18, 7))
    fig5.suptitle(f'声发射事件空间分布 ({len(evts)} 事件)', fontsize=12, fontweight='bold')
    sk = dict(c=evts['time'], cmap='plasma', s=4, alpha=0.5)
    axes5[0].scatter(evts['x'], evts['y'], **sk)
    axes5[0].set_xlabel('x (mm)'); axes5[0].set_ylabel('y (mm)'); axes5[0].set_title('XY')
    sc = axes5[1].scatter(evts['x'], evts['z'], **sk)
    axes5[1].set_xlabel('x (mm)'); axes5[1].set_ylabel('z (mm)'); axes5[1].set_title('XZ')
    axes5[2].scatter(evts['y'], evts['z'], **sk)
    axes5[2].set_xlabel('y (mm)'); axes5[2].set_ylabel('z (mm)'); axes5[2].set_title('YZ')
    plt.colorbar(sc, ax=axes5[2], label='时间 (s)', shrink=0.8)
    plt.tight_layout()
    fig5.savefig(os.path.join(STEP1_DIR, 'fig5_AE事件空间分布.png'))
    plt.close(fig5)
    print(f"  图5 AE 事件空间分布")

print(f"\n{'='*60}")
print(f"Step 1 完成  →  输出目录: {STEP1_DIR}")
print(f"  原始 {n_hits} hits  →  保留 {n_clean} ({100-pct:.1f}%)  去除 {n_contam} ({pct:.1f}%)")
print(f"  下一步: 运行 step2_manual_clean.py 进行手动精细清理")
