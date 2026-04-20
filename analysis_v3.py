#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花岗岩单轴压缩试验 - 改进版综合分析 v3
改进点：
1. Vp = H / (t_AIC_test − t_AIC_cal)   对零波形AIC作为系统延时参考
2. 组合干扰识别：时间窗 ＋ 振幅特征（~60dB连续带 = 超声波干扰）
3. 结果保存至新建"结果"子目录

Sample: Granite Φ50mm × H100mm  |  Date: 2026-04-15
"""

import sys, io
# 强制 stdout 使用 UTF-8（避免 Windows GBK 终端 UnicodeEncodeError）
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
import warnings, os, gc

warnings.filterwarnings('ignore')

# ─── 中文字体 ────────────────────────────────────────────────────────────────
for _f in ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams['font.family'] = _f
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                     'legend.fontsize': 9, 'axes.grid': True, 'grid.alpha': 0.25,
                     'grid.linewidth': 0.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight'})

# ─── 路径 ────────────────────────────────────────────────────────────────────
BASE     = r'g:\Cursor project\ZCY-shengfashe'
US_FILE  = os.path.join(BASE, '超声波', '04-15 - ultrasonics data.csv')
CAL_FILE = os.path.join(BASE, '超声波', 'chushi.csv')
AE_HITS  = os.path.join(BASE, '声发射', '04-15-hits-振铃计数、能量等.TXT')
AE_EVTS  = os.path.join(BASE, '声发射', '04-15-声发射事件.TXT')
AE_CUM_FILES = [os.path.join(BASE, '声发射', f'04-15-hit-累计撞击数-{i}.TXT')
                for i in range(1, 7)]

RESULT_DIR = os.path.join(BASE, '结果')
os.makedirs(RESULT_DIR, exist_ok=True)

# ─── 参数 ────────────────────────────────────────────────────────────────────
H_MM  = 100.0         # 样品高度 mm
H_M   = H_MM / 1000.0
FS_HZ = 40e6          # 采样率 Hz
DT_US = 1e6 / FS_HZ  # 0.025 μs/sample
# 带通滤波
BP_LOW_HZ  = 50e3
BP_HIGH_HZ = 700e3
BP_ORDER   = 4
# AIC搜索窗 (μs)
AIC_SEARCH_OFFSET_US = 12.0   # 搜索起点 = 软件值 - offset
AIC_SEARCH_WIDTH_US  = 30.0   # 搜索宽度
AIC_GLOBAL_START_US  = 5.0    # 绝对下限（避免电气脉冲）
# 时间窗干扰屏蔽 (s)
US_MASK_PRE  = 0.05   # 激发前 50 ms
US_MASK_POST = 0.50   # 激发后 500 ms（基础时间窗）
# 振幅扩展窗 (s) — 对~60dB干扰频带使用更长窗
US_MASK_AMP_LONG = 2.0
# 颜色
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
CH_COLORS = {i + 1: COLORS[i] for i in range(6)}


# ═══════════════════════════════════════════════════════════════════════════
# § 0  辅助函数
# ═══════════════════════════════════════════════════════════════════════════
def butter_bandpass_sos(fs, low, high, order=4):
    nyq = fs / 2
    return butter(order, [low / nyq, high / nyq], btype='band', output='sos')


def apply_bp(signal, sos):
    try:
        return sosfiltfilt(sos, signal.astype(float))
    except Exception:
        return signal.astype(float)


def aic_pick(waveform, t_us_arr, search_start_us, search_end_us):
    """
    AIC P波初至拾取 (O(N) 向量化版本)
    返回: (初至时间μs, 平滑AIC曲线, 搜索索引范围)
    """
    N = len(waveform)
    i0 = max(1, int(np.searchsorted(t_us_arr, search_start_us)))
    i1 = min(N - 1, int(np.searchsorted(t_us_arr, search_end_us)))
    if i1 <= i0 + 2:
        return np.nan, None, (i0, i1)

    x = waveform - waveform.mean()
    cumsum = np.cumsum(x * x)
    total  = cumsum[-1] + 1e-30

    k    = np.arange(i0, i1)
    var1 = cumsum[k - 1] / k
    var2 = (total - cumsum[k - 1]) / (N - k)
    aic  = k * np.log(np.maximum(var1, 1e-30)) + (N - k) * np.log(np.maximum(var2, 1e-30))
    aic_s = np.convolve(aic, np.ones(5) / 5, mode='same')
    idx_min = k[np.argmin(aic_s)]
    return t_us_arr[idx_min], aic_s, (i0, i1)


# ═══════════════════════════════════════════════════════════════════════════
# § 1  加载超声波波形数据
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("加载超声波测试数据...")

us_raw  = pd.read_csv(US_FILE, header=None, low_memory=False, dtype=str)
us_ts   = pd.to_numeric(us_raw.iloc[2, 1:], errors='coerce').dropna().values
n_sw    = len(us_ts)
us_pt_sw = pd.to_numeric(us_raw.iloc[5, 1:n_sw + 1], errors='coerce').values

print("  解析波形数据（8749 samples × 554 sweeps）...")
wf_block   = us_raw.iloc[7:, :]
wf_time_us = pd.to_numeric(wf_block.iloc[:, 0], errors='coerce').values
wf_data    = wf_block.iloc[:, 1:n_sw + 1].apply(pd.to_numeric, errors='coerce').values
del us_raw, wf_block; gc.collect()

n_samp = wf_data.shape[0]
print(f"  扫描次数: {n_sw}   波形样点数: {n_samp}")
print(f"  时间范围: {us_ts[0]:.1f} – {us_ts[-1]:.1f} s")

# ═══════════════════════════════════════════════════════════════════════════
# § 2  加载对零校准波形并用 AIC 得到系统延时 t_cal
# ═══════════════════════════════════════════════════════════════════════════
print("\n加载校准波形 (chushi.csv)...")
t_cal_aic = np.nan

try:
    cal_raw  = pd.read_csv(CAL_FILE, header=None, skiprows=154,
                           low_memory=False, dtype=str, encoding='latin-1')
    cal_time = pd.to_numeric(cal_raw.iloc[:, 0], errors='coerce').values
    cal_sig  = pd.to_numeric(cal_raw.iloc[:, 1], errors='coerce').values

    valid_cal = ~(np.isnan(cal_time) | np.isnan(cal_sig))
    cal_time  = cal_time[valid_cal]
    cal_sig   = cal_sig[valid_cal]
    print(f"  校准波形: {len(cal_time)} 点  "
          f"振幅范围: [{cal_sig.min() * 1e3:.2f}, {cal_sig.max() * 1e3:.2f}] mV")

    sos_bp = butter_bandpass_sos(FS_HZ, BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER)
    cal_filt = apply_bp(cal_sig, sos_bp)

    # 噪声评估（前100点）
    noise_rms = np.std(cal_sig[:100])
    snr_cal   = np.max(np.abs(cal_sig)) / (noise_rms + 1e-30)
    print(f"  噪声RMS: {noise_rms * 1e3:.3f} mV   SNR: {snr_cal:.1f}")

    if len(cal_time) > 100 and snr_cal > 10:
        # AIC搜索范围：0.5 – 40 μs（对零系统延时通常 < 20 μs）
        t_cal_aic, _, _ = aic_pick(cal_filt, cal_time,
                                   search_start_us=0.5, search_end_us=40.0)
        print(f"  AIC系统延时 t_cal = {t_cal_aic:.3f} μs")
    else:
        print("  警告: 校准波形质量不足，将使用经验估计")

except Exception as e:
    print(f"  警告: 无法读取校准波形 ({e})")
    sos_bp = butter_bandpass_sos(FS_HZ, BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER)
    cal_time = np.array([0.0])
    cal_sig  = np.array([0.0])
    cal_filt = np.array([0.0])

# 系统延时确定
if not np.isnan(t_cal_aic) and t_cal_aic > 0:
    sys_delay = t_cal_aic
    delay_method = f'对零AIC: {sys_delay:.3f} μs'
else:
    # 经验：初期稳定段 Vp_ref = 4800 m/s
    VPR = 4800.0
    early_sw = us_pt_sw[~np.isnan(us_pt_sw)][:20]
    sys_delay = float(np.nanmedian(early_sw) - H_M / VPR * 1e6) if len(early_sw) else 0.0
    delay_method = f'经验估计(Vp_ref={VPR:.0f}m/s): {sys_delay:.3f} μs'

print(f"  系统延时方法: {delay_method}")

# ═══════════════════════════════════════════════════════════════════════════
# § 3  对 554 次扫描执行 AIC 拾取
# ═══════════════════════════════════════════════════════════════════════════
print("\n对554次超声波扫描执行AIC P波拾取...")
t_arr = wf_time_us.copy()

us_pt_aic = np.full(n_sw, np.nan)
for i in range(n_sw):
    wf = wf_data[:, i].astype(float)
    if np.sum(np.isnan(wf)) > n_samp * 0.5:
        continue
    wf[np.isnan(wf)] = 0.0
    wf_f = apply_bp(wf, sos_bp)
    sw_ref  = us_pt_sw[i] if not np.isnan(us_pt_sw[i]) else 30.0
    s_start = max(AIC_GLOBAL_START_US, sw_ref - AIC_SEARCH_OFFSET_US)
    s_end   = sw_ref + AIC_SEARCH_WIDTH_US
    us_pt_aic[i], _, _ = aic_pick(wf_f, t_arr, s_start, s_end)

valid_aic = ~np.isnan(us_pt_aic)
print(f"  成功拾取: {valid_aic.sum()} / {n_sw}")
print(f"  AIC到时: {np.nanmin(us_pt_aic):.2f} – {np.nanmax(us_pt_aic):.2f} μs")
print(f"  软件到时: {np.nanmin(us_pt_sw):.2f} – {np.nanmax(us_pt_sw):.2f} μs")

diff = us_pt_aic - us_pt_sw
diff_valid = diff[valid_aic & ~np.isnan(us_pt_sw)]
print(f"  AIC-软件: 均值={np.nanmean(diff_valid):.3f} us  sigma={np.nanstd(diff_valid):.3f} us")

# ─── Vp 计算 (对零AIC校正) ─────────────────────────────────────────────────
# Vp = H / (t_AIC_test − t_cal)   [t 单位: μs → 换算 s; Vp: m/s → km/s]
travel_aic = (us_pt_aic - sys_delay) * 1e-6   # seconds
travel_sw  = (us_pt_sw  - sys_delay) * 1e-6

us_Vp_aic = np.where((travel_aic > 5e-6) & (travel_aic < 200e-6),
                      H_M / travel_aic / 1000.0, np.nan)
us_Vp_sw  = np.where((travel_sw  > 5e-6) & (travel_sw  < 200e-6),
                      H_M / travel_sw  / 1000.0, np.nan)

print(f"\n  Vp(AIC, 对零校正): {np.nanmin(us_Vp_aic):.2f} – {np.nanmax(us_Vp_aic):.2f} km/s")
print(f"  Vp(软件, 对零校正): {np.nanmin(us_Vp_sw):.2f} – {np.nanmax(us_Vp_sw):.2f} km/s")

# ═══════════════════════════════════════════════════════════════════════════
# § 4  加载声发射数据
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("加载声发射数据...")


def parse_ae_hits(path):
    rows = []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if (not line or line.startswith('C:') or line.startswith('Express')
                    or line.startswith('Version') or line.startswith('4/')
                    or line.startswith('128') or line.startswith('ID')):
                continue
            parts = line.split()
            if len(parts) >= 9:
                try:
                    rows.append({'Time': float(parts[1]), 'CH': int(parts[2]),
                                 'RISE': int(parts[3]), 'COUN': int(parts[4]),
                                 'ENER': int(parts[5]), 'DURATION': int(parts[6]),
                                 'AMP': float(parts[7]), 'ABS_E': float(parts[8])})
                except Exception:
                    pass
    return pd.DataFrame(rows)


ae   = parse_ae_hits(AE_HITS)
ae   = ae[ae['Time'] > 0].sort_values('Time').reset_index(drop=True)
ae_t = ae['Time'].values
ae_amp = ae['AMP'].values
print(f"  撞击总数: {len(ae)}   时间: {ae_t.min():.2f} – {ae_t.max():.2f} s")


def parse_ae_events(path):
    events = []
    with open(path, 'r', errors='replace') as fh:
        current = None
        for line in fh:
            line = line.strip()
            if line.startswith('* Gp#'):
                try:
                    x = float(line.split('x,y,z =')[1].split(',')[0])
                    y = float(line.split('x,y,z =')[1].split(',')[1])
                    z = float(line.split('x,y,z =')[1].split(',')[2].split(',')[0])
                    q = float(line.split('q =')[1].strip().split()[0])
                    sa = float(line.split('Src Amplitude =')[1].strip().split()[0]) \
                        if 'Src Amplitude' in line else 0.0
                    current = {'x': x, 'y': y, 'z': z, 'q': q, 'src_amp': sa, 'time': None}
                    events.append(current)
                except Exception:
                    current = None
            elif line.startswith('*') and current is not None:
                parts = line.lstrip('*').split()
                if len(parts) >= 8:
                    try:
                        t = float(parts[0])
                        if current['time'] is None:
                            current['time'] = t
                    except ValueError:
                        pass
    return pd.DataFrame([e for e in events if e['time'] is not None])


evts = parse_ae_events(AE_EVTS)
print(f"  已定位事件: {len(evts)}")

cum_dfs = []
for i, f in enumerate(AE_CUM_FILES):
    try:
        df = pd.read_csv(f, sep=r'\s+', skiprows=1,
                         names=['Time', f'CH{i + 1}'], engine='python')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        cum_dfs.append(df)
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════
# § 5  组合干扰识别
#   方法：
#     (A) 基础时间窗：US激发前50ms / 后500ms 内的所有撞击
#     (B) 振幅扩展窗：US激发前50ms / 后2000ms 内，且振幅落在~60dB干扰频带内的撞击
#   最终干扰 = (A) 或 (B)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("组合干扰识别（时间窗 + 振幅特征）...")

period_approx = float(np.median(np.diff(us_ts)))
print(f"  US脉冲平均间隔: {period_approx:.3f} s")

# ── 步骤1：先用基础时间窗标记干扰撞击，用于确定干扰振幅频带 ──────────────
contam_time_only = np.zeros(len(ae_t), dtype=bool)
for t_us in us_ts:
    contam_time_only |= (ae_t >= t_us - US_MASK_PRE) & (ae_t <= t_us + US_MASK_POST)

# 用早期（t < 200s）时间窗内的撞击拟合干扰振幅分布
early_window_mask = contam_time_only & (ae_t < 200.0)
amp_interference  = ae_amp[early_window_mask]

if len(amp_interference) >= 30:
    amp_edges   = np.arange(35, 125, 2)           # 2dB bins
    amp_hist, _ = np.histogram(amp_interference, bins=amp_edges)
    amp_centers = (amp_edges[:-1] + amp_edges[1:]) / 2
    peak_idx    = int(np.argmax(amp_hist))
    amp_peak    = float(amp_centers[peak_idx])
    # 以峰值为中心，±8 dB 为干扰频带
    AMP_US_HALF_WIDTH = 8.0
    AMP_US_LOW  = amp_peak - AMP_US_HALF_WIDTH
    AMP_US_HIGH = amp_peak + AMP_US_HALF_WIDTH
    print(f"  从 {len(amp_interference)} 个早期时间窗撞击中拟合干扰振幅")
    print(f"  干扰振幅峰值: {amp_peak:.1f} dB  →  频带: [{AMP_US_LOW:.1f}, {AMP_US_HIGH:.1f}] dB")
else:
    # 根据用户观察直接设定 ~60dB
    AMP_US_LOW, AMP_US_HIGH = 52.0, 68.0
    amp_peak = 60.0
    print(f"  早期数据不足，使用默认干扰频带: [{AMP_US_LOW:.1f}, {AMP_US_HIGH:.1f}] dB")

# ── 步骤2：计算最终组合干扰标记 ────────────────────────────────────────────
contam_base = np.zeros(len(ae_t), dtype=bool)   # 基础时间窗 (A)
contam_long = np.zeros(len(ae_t), dtype=bool)   # 扩展振幅窗 (B，待与振幅条件取交集)

for t_us in us_ts:
    contam_base |= (ae_t >= t_us - US_MASK_PRE) & (ae_t <= t_us + US_MASK_POST)
    contam_long |= (ae_t >= t_us - US_MASK_PRE) & (ae_t <= t_us + US_MASK_AMP_LONG)

# (B) = 扩展时间窗 AND 振幅在干扰频带内
in_amp_band = (ae_amp >= AMP_US_LOW) & (ae_amp <= AMP_US_HIGH)
contam_amp_windowed = contam_long & in_amp_band

# 最终干扰 = (A) 或 (B)
contam = contam_base | contam_amp_windowed

ae_clean  = ae[~contam].reset_index(drop=True)
ae_contam = ae[contam].reset_index(drop=True)
n_total   = len(ae)
n_contam  = int(contam.sum())
n_clean   = n_total - n_contam
pct       = 100.0 * n_contam / n_total

# 各滤波器单独统计
n_base_only = int(contam_base.sum())
n_amp_extra = int((contam_amp_windowed & ~contam_base).sum())
print(f"  基础时间窗去除: {n_base_only} hits  ({100.*n_base_only/n_total:.1f}%)")
print(f"  振幅扩展窗额外去除: {n_amp_extra} hits  ({100.*n_amp_extra/n_total:.1f}%)")
print(f"  原始: {n_total}  →  干扰: {n_contam} ({pct:.1f}%)  →  真实: {n_clean} ({100-pct:.1f}%)")

# ── 破坏时刻检测 ──────────────────────────────────────────────────────────
vp_base    = np.nanmedian(us_Vp_aic[valid_aic][:30]) if valid_aic.sum() >= 30 else np.nan
fail_mask  = (~np.isnan(us_Vp_aic)) & (us_Vp_aic < (vp_base * 0.7)) if not np.isnan(vp_base) else np.zeros(n_sw, dtype=bool)
t_fail     = float(us_ts[np.where(fail_mask)[0][0]]) if np.any(fail_mask) else np.nan

# ═══════════════════════════════════════════════════════════════════════════
# § 6  绘图
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("生成图表...")


def mark_failure(axes_list, t_f):
    if not np.isnan(t_f):
        for ax in axes_list:
            ax.axvline(t_f, color='red', ls='--', lw=1.2, alpha=0.7)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图1  对零校准波形 + 典型测试波形 + AIC拾取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig1 = plt.figure(figsize=(16, 10))
gs1  = gridspec.GridSpec(2, 4, figure=fig1, hspace=0.45, wspace=0.38)
fig1.suptitle('AIC P波拾取示例  |  对零校准 + 典型测试波形', fontsize=12, fontweight='bold')

ax_cal = fig1.add_subplot(gs1[0, :2])
if len(cal_time) > 100:
    t_mask_cal = cal_time <= 60
    ax_cal.plot(cal_time[t_mask_cal], cal_sig[t_mask_cal] * 1000,
                color='gray', lw=0.8, alpha=0.6, label='原始')
    ax_cal.plot(cal_time[t_mask_cal], cal_filt[t_mask_cal] * 1000,
                color='steelblue', lw=1.2, label='带通滤波后')
    if not np.isnan(t_cal_aic):
        ax_cal.axvline(t_cal_aic, color='red', lw=1.8, ls='--',
                       label=f'AIC t_cal = {t_cal_aic:.3f} μs\n(系统延时)')
ax_cal.set_xlabel('时间 (μs)')
ax_cal.set_ylabel('振幅 (mV)')
ax_cal.set_title(f'对零校准波形 (系统延时 = {sys_delay:.3f} μs)')
ax_cal.legend(fontsize=8)

# 拾取差值直方图
ax_diff = fig1.add_subplot(gs1[0, 2:])
d = diff_valid[np.abs(diff_valid) < 10]
ax_diff.hist(d, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
ax_diff.axvline(np.nanmean(d), color='red', lw=1.5, ls='--',
                label=f'均值 {np.nanmean(d):.2f} μs')
ax_diff.set_xlabel('AIC到时 − 软件到时 (μs)')
ax_diff.set_ylabel('频数')
ax_diff.set_title('拾取差值分布（AIC vs 软件）')
ax_diff.legend()

test_times  = [us_ts[5], us_ts[len(us_ts) // 4], us_ts[len(us_ts) * 3 // 4], us_ts[-20]]
test_labels = ['初期', '1/4段', '3/4段', '临近破坏']
test_colors = ['steelblue', 'green', 'orange', 'red']

for col_idx, (t_show, lbl, clr) in enumerate(zip(test_times, test_labels, test_colors)):
    sw_idx = int(np.argmin(np.abs(us_ts - t_show)))
    ax_t   = fig1.add_subplot(gs1[1, col_idx])
    wf_raw = wf_data[:, sw_idx].astype(float)
    wf_raw[np.isnan(wf_raw)] = 0.0
    wf_f   = apply_bp(wf_raw, sos_bp)
    sw_ref  = us_pt_sw[sw_idx] if not np.isnan(us_pt_sw[sw_idx]) else 30.0
    s_start = max(AIC_GLOBAL_START_US, sw_ref - AIC_SEARCH_OFFSET_US)
    s_end   = sw_ref + AIC_SEARCH_WIDTH_US
    t_aic_i, aic_curve, (i0, i1) = aic_pick(wf_f, t_arr, s_start, s_end)
    plot_mask = t_arr <= 120
    ax_t.plot(t_arr[plot_mask], wf_raw[plot_mask] * 1e3, color='lightgray', lw=0.7, label='原始')
    ax_t.plot(t_arr[plot_mask], wf_f[plot_mask] * 1e3, color=clr, lw=1.0, alpha=0.9, label='滤波后')
    if aic_curve is not None:
        aic_t  = t_arr[i0:i1]
        pa     = aic_t <= 120
        an     = aic_curve[pa] - aic_curve[pa].min()
        an_max = an.max() + 1e-30
        an     = an / an_max * np.abs(wf_f[plot_mask]).max() * 1e3
        ax_t.plot(aic_t[pa], -an, color='purple', lw=0.8, ls=':', alpha=0.7, label='−AIC')
    if not np.isnan(t_aic_i):
        ax_t.axvline(t_aic_i, color='red', lw=1.5, ls='--', label=f'AIC={t_aic_i:.2f}μs')
    if not np.isnan(us_pt_sw[sw_idx]):
        ax_t.axvline(us_pt_sw[sw_idx], color='navy', lw=1.2, ls=':', label=f'软件={us_pt_sw[sw_idx]:.2f}μs')
    ax_t.set_xlabel('时间 (μs)')
    ax_t.set_ylabel('振幅 (mV)')
    ax_t.set_title(f'{lbl}  t={t_show:.0f}s')
    ax_t.legend(fontsize=7, ncol=2)
    ax_t.set_xlim(0, 120)

out1 = os.path.join(RESULT_DIR, 'v3_01_AIC波形拾取示例.png')
fig1.savefig(out1)
plt.close(fig1)
print(f"图1已保存: {out1}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图2  P波到时与速度 (对零AIC校正后)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), sharex='col')
fig2.suptitle(f'P波速度演化 (对零AIC校正)  —  系统延时 t_cal = {sys_delay:.3f} μs',
              fontsize=12, fontweight='bold')

(ax2a, ax2b), (ax2c, ax2d) = axes2

ax2a.plot(us_ts, us_pt_sw,  color='navy',    lw=0.8, alpha=0.8, label='软件自动拾取')
ax2a.plot(us_ts, us_pt_aic, color='crimson', lw=0.8, alpha=0.8, label='AIC拾取')
ax2a.set_ylabel('P波到时 (μs)')
ax2a.set_title('P波到时（全段）')
ax2a.legend()
ax2a.yaxis.set_minor_locator(AutoMinorLocator())
mark_failure([ax2a], t_fail)

ax2b.plot(us_ts, diff, color='purple', lw=0.6, alpha=0.7)
ax2b.axhline(0, color='black', lw=0.8, ls='--')
ax2b.axhline(np.nanmean(diff_valid), color='red', lw=1, ls=':',
             label=f'均值 {np.nanmean(diff_valid):.2f} μs')
ax2b.set_ylabel('AIC − 软件 (μs)')
ax2b.set_title('拾取差值')
ax2b.legend()

valid_both = ~(np.isnan(us_Vp_aic) | np.isnan(us_Vp_sw))
ax2c.plot(us_ts[valid_both], us_Vp_sw[valid_both],
          color='navy',    lw=0.8, alpha=0.8, label='Vp (软件)')
ax2c.plot(us_ts[valid_both], us_Vp_aic[valid_both],
          color='crimson', lw=0.8, alpha=0.8, label='Vp (AIC)')
ax2c.set_ylabel('P波速度 (km/s)')
ax2c.set_xlabel('时间 (s)')
ax2c.set_title(f'P波速度演化  (Vp = {H_MM:.0f}mm / (t_AIC − {sys_delay:.3f}μs))')
ax2c.legend()
ax2c.yaxis.set_minor_locator(AutoMinorLocator())
mark_failure([ax2a, ax2c], t_fail)

ax2d.scatter(us_Vp_sw[valid_both], us_Vp_aic[valid_both], s=3, alpha=0.4, color='steelblue')
lim = [min(np.nanmin(us_Vp_sw[valid_both]), np.nanmin(us_Vp_aic[valid_both])) * 0.95,
       max(np.nanmax(us_Vp_sw[valid_both]), np.nanmax(us_Vp_aic[valid_both])) * 1.05]
ax2d.plot(lim, lim, 'r--', lw=1, label='1:1')
ax2d.set_xlabel('Vp 软件 (km/s)')
ax2d.set_ylabel('Vp AIC (km/s)')
ax2d.set_title('速度对比散点图')
ax2d.legend()
ax2d.set_xlim(lim); ax2d.set_ylim(lim)

plt.tight_layout()
out2 = os.path.join(RESULT_DIR, 'v3_02_P波速度AIC对比.png')
fig2.savefig(out2)
plt.close(fig2)
print(f"图2已保存: {out2}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图3  干扰振幅分析：早期振幅分布 + 干扰频带确定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle('超声波干扰振幅特征分析\n'
              'Interference Amplitude Characterization (Early-stage US-window hits)',
              fontsize=12, fontweight='bold')

# 早期时间窗内撞击的振幅分布
if len(amp_interference) >= 30:
    axes3[0].bar(amp_centers, amp_hist, width=2 * 0.9,
                 color='tomato', alpha=0.7, label=f'早期时间窗内撞击 (n={len(amp_interference)})')
axes3[0].axvline(AMP_US_LOW,  color='navy', lw=1.5, ls='--', label=f'频带 {AMP_US_LOW:.0f}–{AMP_US_HIGH:.0f} dB')
axes3[0].axvline(AMP_US_HIGH, color='navy', lw=1.5, ls='--')
axes3[0].axvspan(AMP_US_LOW, AMP_US_HIGH, alpha=0.12, color='navy')
axes3[0].set_xlabel('振幅 (dB)')
axes3[0].set_ylabel('频数')
axes3[0].set_title('早期超声波干扰振幅分布\n(t < 200 s, 时间窗内撞击)')
axes3[0].legend(fontsize=8)

# 全实验振幅直方图（干扰 vs 真实）
amp_edges_all = np.arange(35, 105, 2)
amp_ctrs_all  = (amp_edges_all[:-1] + amp_edges_all[1:]) / 2
h_contam, _   = np.histogram(ae_amp[contam],  bins=amp_edges_all)
h_clean,  _   = np.histogram(ae_amp[~contam], bins=amp_edges_all)
axes3[1].bar(amp_ctrs_all, h_contam, width=2 * 0.9, color='silver', alpha=0.8, label=f'标记为干扰 ({n_contam})')
axes3[1].bar(amp_ctrs_all, h_clean,  width=2 * 0.9, color='steelblue', alpha=0.7,
             bottom=h_contam, label=f'真实AE ({n_clean})')
axes3[1].axvline(AMP_US_LOW,  color='red', lw=1.5, ls='--')
axes3[1].axvline(AMP_US_HIGH, color='red', lw=1.5, ls='--', label=f'干扰频带 [{AMP_US_LOW:.0f},{AMP_US_HIGH:.0f}]dB')
axes3[1].axvspan(AMP_US_LOW, AMP_US_HIGH, alpha=0.10, color='red')
axes3[1].set_xlabel('振幅 (dB)')
axes3[1].set_ylabel('频数')
axes3[1].set_title('全段振幅分布（干扰 vs 真实AE）')
axes3[1].legend(fontsize=8)

# 振幅频带内撞击的时间分布（展示规律性）
band_mask = in_amp_band
ae_t_band = ae_t[band_mask]
TBIN = 10.0
t_edges = np.arange(0, ae_t.max() + TBIN, TBIN)
h_band, _ = np.histogram(ae_t_band, bins=t_edges)
h_all,  _ = np.histogram(ae_t,      bins=t_edges)
t_ctrs_hist = (t_edges[:-1] + t_edges[1:]) / 2
axes3[2].plot(t_ctrs_hist, h_band, color='tomato', lw=1.0, alpha=0.8,
              label=f'振幅频带 [{AMP_US_LOW:.0f},{AMP_US_HIGH:.0f}]dB 内撞击')
axes3[2].plot(t_ctrs_hist, h_all,  color='steelblue', lw=1.0, alpha=0.5, label='全部撞击')
if not np.isnan(t_fail):
    axes3[2].axvline(t_fail, color='red', ls='--', lw=1.2, label=f'破坏 t={t_fail:.0f}s')
axes3[2].set_xlabel('时间 (s)')
axes3[2].set_ylabel(f'撞击数 / {TBIN:.0f}s bin')
axes3[2].set_title('干扰振幅频带内撞击的时间分布')
axes3[2].legend(fontsize=8)

plt.tight_layout()
out3 = os.path.join(RESULT_DIR, 'v3_03_干扰振幅分析.png')
fig3.savefig(out3)
plt.close(fig3)
print(f"图3已保存: {out3}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图4  全段干扰对比图（6通道，清晰标注~60dB干扰频带）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("生成全段干扰对比图（6通道）...")

T_MAX      = ae_t.max() + 20
ALPHA_DOT  = 0.40
SZ         = 1.5
AMP_Y_LIM  = [ae_amp.min() - 5, ae_amp.max() + 5]

fig4 = plt.figure(figsize=(20, 28))
gs4  = gridspec.GridSpec(6, 2, figure=fig4,
                         hspace=0.10, wspace=0.06,
                         top=0.94, bottom=0.04, left=0.07, right=0.97)
fig4.suptitle(
    f'声发射振幅 — 超声波干扰去除 全段对比图 (0 – {T_MAX:.0f} s)\n'
    f'干扰识别: 时间窗(前{US_MASK_PRE*1000:.0f}ms/后{US_MASK_POST*1000:.0f}ms) 或 '
    f'振幅频带[{AMP_US_LOW:.0f},{AMP_US_HIGH:.0f}]dB内(扩展窗{US_MASK_AMP_LONG*1000:.0f}ms)\n'
    f'去除 {n_contam} hits ({pct:.1f}%),  保留 {n_clean} 真实AE hits ({100-pct:.1f}%)',
    fontsize=11, fontweight='bold')

from matplotlib.lines import Line2D

for ch in range(1, 7):
    row = ch - 1
    d_ct = ae_contam[ae_contam['CH'] == ch]
    d_cl = ae_clean[ae_clean['CH'] == ch]

    # ── 左列：原始数据（干扰灰色，真实彩色）
    axL = fig4.add_subplot(gs4[row, 0])
    if len(d_ct):
        axL.scatter(d_ct['Time'], d_ct['AMP'],
                    s=SZ, alpha=ALPHA_DOT * 0.5, color='silver', zorder=1, rasterized=True)
    if len(d_cl):
        axL.scatter(d_cl['Time'], d_cl['AMP'],
                    s=SZ, alpha=ALPHA_DOT, color=CH_COLORS[ch], zorder=2, rasterized=True)
    # 干扰振幅频带参考线
    axL.axhspan(AMP_US_LOW, AMP_US_HIGH, alpha=0.08, color='red', zorder=0)
    axL.axhline(AMP_US_LOW,  color='red', lw=0.7, ls=':', alpha=0.6)
    axL.axhline(AMP_US_HIGH, color='red', lw=0.7, ls=':', alpha=0.6)
    axL.text(T_MAX * 0.01, (AMP_US_LOW + AMP_US_HIGH) / 2,
             f'US干扰带\n{AMP_US_LOW:.0f}–{AMP_US_HIGH:.0f}dB',
             fontsize=6, color='red', va='center', alpha=0.7)
    axL.set_ylabel(f'CH{ch}\n振幅(dB)', fontsize=9)
    axL.set_ylim(AMP_Y_LIM)
    axL.set_xlim(0, T_MAX)
    axL.yaxis.set_major_locator(MultipleLocator(20))
    axL.yaxis.set_minor_locator(MultipleLocator(10))
    if row == 0:
        axL.set_title(f'原始数据 (共 {n_total} hits)', fontsize=11, pad=8)
        leg_h = [Line2D([0], [0], marker='o', ls='None', color='silver',
                        markersize=4, label=f'超声波干扰 ({n_contam}, {pct:.1f}%)'),
                 Line2D([0], [0], marker='o', ls='None', color=CH_COLORS[ch],
                        markersize=4, label=f'真实AE ({n_clean}, {100-pct:.1f}%)')]
        axL.legend(handles=leg_h, fontsize=8, loc='upper left')

    # ── 右列：去干扰后数据
    axR = fig4.add_subplot(gs4[row, 1], sharey=axL)
    if len(d_cl):
        axR.scatter(d_cl['Time'], d_cl['AMP'],
                    s=SZ, alpha=ALPHA_DOT, color=CH_COLORS[ch], rasterized=True)
    # 参考线
    axR.axhspan(AMP_US_LOW, AMP_US_HIGH, alpha=0.05, color='red', zorder=0)
    axR.axhline(AMP_US_LOW,  color='red', lw=0.6, ls=':', alpha=0.4)
    axR.axhline(AMP_US_HIGH, color='red', lw=0.6, ls=':', alpha=0.4)
    axR.set_xlim(0, T_MAX)
    axR.set_ylim(AMP_Y_LIM)
    axR.yaxis.set_major_locator(MultipleLocator(20))
    axR.tick_params(labelleft=False)
    if row == 0:
        axR.set_title(f'去超声波干扰后  ({n_clean} hits, {100-pct:.1f}%)', fontsize=11, pad=8)

    if row < 5:
        axL.tick_params(labelbottom=False)
        axR.tick_params(labelbottom=False)
    else:
        axL.set_xlabel('时间 (s)')
        axR.set_xlabel('时间 (s)')

    if not np.isnan(t_fail):
        axL.axvline(t_fail, color='red', lw=0.8, ls=':', alpha=0.6)
        axR.axvline(t_fail, color='red', lw=0.8, ls=':', alpha=0.6)

out4 = os.path.join(RESULT_DIR, 'v3_04_全段干扰对比图.png')
fig4.savefig(out4)
plt.close(fig4)
print(f"图4已保存: {out4}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图5  综合分析（AIC Vp + AE去干扰后 + 累计撞击）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig5, axes5 = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
fig5.suptitle('超声波与声发射综合分析 (v3)\n'
              'Combined Analysis: Ultrasonic Vp (AIC, calibrated) + AE (interference removed)',
              fontsize=12, fontweight='bold')
ax5a, ax5b, ax5c, ax5d = axes5

m_vp = ~np.isnan(us_Vp_aic)
ax5a.plot(us_ts[m_vp], us_Vp_aic[m_vp], color='crimson', lw=1, alpha=0.85, label='Vp AIC (对零校正)')
ax5a.fill_between(us_ts[m_vp], us_Vp_aic[m_vp], alpha=0.12, color='crimson')
ax5a.set_ylabel('Vp (km/s)')
ax5a.set_title(f'P波速度演化 (AIC法, Vp = {H_MM:.0f}mm / (t_AIC − t_cal))')
ax5a.yaxis.set_minor_locator(AutoMinorLocator())
ax5a.legend(loc='upper left')

for ch in range(1, 7):
    d = ae_clean[ae_clean['CH'] == ch]
    ax5b.scatter(d['Time'], d['AMP'], s=1.5, alpha=0.35,
                 color=CH_COLORS[ch], label=f'CH{ch}', rasterized=True)
ax5b.axhspan(AMP_US_LOW, AMP_US_HIGH, alpha=0.08, color='red')
ax5b.axhline(AMP_US_LOW,  color='red', lw=0.8, ls=':', alpha=0.5)
ax5b.axhline(AMP_US_HIGH, color='red', lw=0.8, ls=':', alpha=0.5, label=f'原干扰带 {AMP_US_LOW:.0f}–{AMP_US_HIGH:.0f}dB')
ax5b.set_ylabel('AE振幅 (dB)')
ax5b.set_title('声发射振幅（组合干扰去除后）')
h = [plt.Line2D([0], [0], marker='o', ls='None', color=CH_COLORS[i + 1],
                markersize=5, label=f'CH{i + 1}') for i in range(6)]
ax5b.legend(handles=h, ncol=6, fontsize=8, loc='upper left', framealpha=0.6)

ax5c.scatter(ae_clean['Time'], ae_clean['ABS_E'], s=1.5, alpha=0.3,
             color='darkred', rasterized=True)
ax5c.set_yscale('log')
ax5c.set_ylabel('绝对能量 (aJ)')
ax5c.set_title('声发射绝对能量（去干扰后）')

ae_cl_s = ae_clean.sort_values('Time')
ae_or_s = ae.sort_values('Time')
ax5d.plot(ae_or_s['Time'], np.arange(1, n_total + 1),
          color='tomato', lw=1.2, ls='--', alpha=0.7, label=f'原始 ({n_total})')
ax5d.plot(ae_cl_s['Time'], np.arange(1, n_clean + 1),
          color='steelblue', lw=1.5, alpha=0.9, label=f'去干扰后 ({n_clean})')
ax5d.set_ylabel('累计撞击数')
ax5d.set_xlabel('时间 (s)')
ax5d.set_title('累计撞击数对比')
ax5d.legend()

mark_failure(axes5, t_fail)
if not np.isnan(t_fail):
    axes5[0].text(t_fail + 15, ax5a.get_ylim()[0] * 1.02,
                  f'破坏 t={t_fail:.0f}s', color='red', fontsize=9)

plt.tight_layout()
out5 = os.path.join(RESULT_DIR, 'v3_05_综合分析.png')
fig5.savefig(out5)
plt.close(fig5)
print(f"图5已保存: {out5}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图6  声发射事件空间分布
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if len(evts) > 0 and 'x' in evts.columns:
    fig6, axes6 = plt.subplots(1, 3, figsize=(18, 7))
    fig6.suptitle(f'声发射事件空间分布  (已定位 {len(evts)} 个事件)\n'
                  'AE Event Spatial Distribution (Located Events)',
                  fontsize=12, fontweight='bold')
    scatter_kw = dict(c=evts['time'], cmap='plasma', s=4, alpha=0.5)
    axes6[0].scatter(evts['x'], evts['y'], **scatter_kw)
    axes6[0].set_xlabel('x (mm)'); axes6[0].set_ylabel('y (mm)')
    axes6[0].set_title('XY 平面')
    sc = axes6[1].scatter(evts['x'], evts['z'], **scatter_kw)
    axes6[1].set_xlabel('x (mm)'); axes6[1].set_ylabel('z (mm)')
    axes6[1].set_title('XZ 平面')
    axes6[2].scatter(evts['y'], evts['z'], **scatter_kw)
    axes6[2].set_xlabel('y (mm)'); axes6[2].set_ylabel('z (mm)')
    axes6[2].set_title('YZ 平面')
    plt.colorbar(sc, ax=axes6[2], label='时间 (s)', shrink=0.8)
    # 加载方向轴线 (y轴为高度方向0-100mm)
    for ax in axes6[:2]:
        ax.axhline(0,  color='gray', lw=0.8, ls='--', alpha=0.4)
        ax.axhline(100, color='gray', lw=0.8, ls='--', alpha=0.4)
    plt.tight_layout()
    out6 = os.path.join(RESULT_DIR, 'v3_06_AE事件空间分布.png')
    fig6.savefig(out6)
    plt.close(fig6)
    print(f"图6已保存: {out6}")
else:
    out6 = None
    print("图6: 事件数据不足，跳过")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 保存 CSV 结果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df_vp = pd.DataFrame({
    'time_s':       us_ts,
    'pwave_sw_us':  us_pt_sw,
    'pwave_aic_us': us_pt_aic,
    'travel_aic_us': (us_pt_aic - sys_delay),
    'Vp_sw_km':     us_Vp_sw,
    'Vp_aic_km':    us_Vp_aic,
})
out_vp = os.path.join(RESULT_DIR, 'v3_Vp_AIC.csv')
df_vp.to_csv(out_vp, index=False)

out_ae = os.path.join(RESULT_DIR, 'v3_AE_clean.csv')
ae_clean.to_csv(out_ae, index=False)

out_ae_contam = os.path.join(RESULT_DIR, 'v3_AE_contaminated.csv')
ae_contam.to_csv(out_ae_contam, index=False)

# ─── 汇总统计 ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("统计汇总 | Summary")
print("=" * 60)
print(f"\n[超声波 P波速度 (AIC 对零校正)]")
print(f"  系统延时 t_cal:        {sys_delay:.3f} μs  ({delay_method})")
print(f"  校正方法:              Vp = {H_MM:.0f}mm / (t_AIC - t_cal)")
_vp_early = us_Vp_aic[valid_aic][:10]
print(f"  初始 Vp (前10次):      {np.nanmean(_vp_early):.3f} km/s" if len(_vp_early) else "  初始 Vp: N/A")
print(f"  峰值 Vp:               {np.nanmax(us_Vp_aic):.3f} km/s")
if not np.isnan(t_fail):
    pre_fail_idx = np.where(fail_mask)[0][0]
    print(f"  破坏前5次 Vp均值:      {np.nanmean(us_Vp_aic[max(0,pre_fail_idx-5):pre_fail_idx]):.3f} km/s")
    print(f"  推测破坏时刻:          {t_fail:.1f} s")
print(f"  AIC vs 软件 差值均值:  {np.nanmean(diff_valid):.3f} μs  (σ={np.nanstd(diff_valid):.3f} μs)")

print(f"\n[声发射干扰识别（组合方法）]")
print(f"  干扰振幅频带:          {AMP_US_LOW:.1f} – {AMP_US_HIGH:.1f} dB  (峰值 {amp_peak:.1f} dB)")
print(f"  基础时间窗:            前 {US_MASK_PRE*1000:.0f}ms / 后 {US_MASK_POST*1000:.0f}ms")
print(f"  振幅扩展时间窗:        前 {US_MASK_PRE*1000:.0f}ms / 后 {US_MASK_AMP_LONG*1000:.0f}ms (仅限干扰频带)")
print(f"  基础时间窗去除:        {n_base_only} hits ({100.*n_base_only/n_total:.1f}%)")
print(f"  振幅扩展窗额外去除:    {n_amp_extra} hits ({100.*n_amp_extra/n_total:.1f}%)")
print(f"  原始撞击数:            {n_total}")
print(f"  干扰撞击数:            {n_contam} ({pct:.1f}%)")
print(f"  真实AE撞击数:          {n_clean} ({100-pct:.1f}%)")

print(f"\n[输出目录]: {RESULT_DIR}")
output_files = [out1, out2, out3, out4, out5, out_vp, out_ae, out_ae_contam]
if out6:
    output_files.append(out6)
for f in output_files:
    print(f"  {os.path.basename(f)}")

print("\n分析完成！")
