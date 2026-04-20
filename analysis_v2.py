#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花岗岩单轴压缩试验 - 改进版综合分析 v2
* AIC方法重新拾取超声波P波到时（含滤波去噪）
* 基于dt分布统计确定超声波干扰时间窗
* 全段（0-1730 s）干扰对比图

Sample: Granite Φ50mm × H100mm  |  Date: 2026-04-15
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.signal import butter, sosfiltfilt
from scipy.stats import linregress
import warnings, os, gc

warnings.filterwarnings('ignore')

# ─── 中文字体 ────────────────────────────────────────────────────────────────
for _f in ['Microsoft YaHei','SimHei','WenQuanYi Micro Hei','Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams['font.family'] = _f
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                     'legend.fontsize':9,'axes.grid':True,'grid.alpha':0.25,
                     'grid.linewidth':0.5,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':150,
                     'savefig.dpi':200,'savefig.bbox':'tight'})

# ─── 路径 ────────────────────────────────────────────────────────────────────
BASE    = r'g:\Cursor project\ZCY-shengfashe'
US_FILE = os.path.join(BASE, '超声波', '04-15 - ultrasonics data.csv')
CAL_FILE= os.path.join(BASE, '超声波', 'chushi.csv')
AE_HITS = os.path.join(BASE, '声发射', '04-15-hits-振铃计数、能量等.TXT')
AE_EVTS = os.path.join(BASE, '声发射', '04-15-声发射事件.TXT')
AE_CUM_FILES = [os.path.join(BASE,'声发射',f'04-15-hit-累计撞击数-{i}.TXT') for i in range(1,7)]

# ─── 参数 ────────────────────────────────────────────────────────────────────
H_MM   = 100.0            # 样品高度 mm
H_M    = H_MM / 1000.0
FS_HZ  = 40e6             # 采样率 Hz
DT_US  = 1e6 / FS_HZ     # 采样间隔 0.025 μs
# 带通滤波参数
BP_LOW_HZ   = 50e3        # 50 kHz
BP_HIGH_HZ  = 700e3       # 700 kHz
BP_ORDER    = 4
# AIC搜索窗（μs）—— 跳过初始电气脉冲, 以软件值为中心扩展搜索
AIC_SEARCH_OFFSET_US = 12.0   # 搜索窗起点：软件值 - offset
AIC_SEARCH_WIDTH_US  = 30.0   # 搜索窗宽度
AIC_GLOBAL_START_US  = 10.0   # 绝对下限（避免电气噪声）
# 颜色
COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
CH_COLORS = {i+1: COLORS[i] for i in range(6)}

# ═══════════════════════════════════════════════════════════════════════════
# § 1  加载超声波波形数据
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("加载超声波测试数据...")

us_raw = pd.read_csv(US_FILE, header=None, low_memory=False, dtype=str)

# 第3行(index 2)：扫描时间戳 (s)
us_ts   = pd.to_numeric(us_raw.iloc[2, 1:], errors='coerce').dropna().values
n_sw    = len(us_ts)

# 第6行(index 5)：软件自动拾取的P波到时 (μs)
us_pt_sw = pd.to_numeric(us_raw.iloc[5, 1:n_sw+1], errors='coerce').values

# 第8行起(index 7+)：波形数据  列0=时间μs  列1~n_sw=电压V
print("  解析波形数据（8749 samples × 554 sweeps）...")
wf_block = us_raw.iloc[7:, :]
wf_time_us = pd.to_numeric(wf_block.iloc[:, 0], errors='coerce').values   # μs
wf_data    = wf_block.iloc[:, 1:n_sw+1].apply(pd.to_numeric, errors='coerce').values  # (N,554)
del us_raw, wf_block; gc.collect()

n_samp = wf_data.shape[0]  # ~8749
print(f"  扫描次数: {n_sw}   波形样点数: {n_samp}")
print(f"  时间范围: {us_ts[0]:.1f} – {us_ts[-1]:.1f} s")
print(f"  波形窗口: 0 – {wf_time_us[-1]:.1f} μs   (采样率 {FS_HZ/1e6:.0f} MHz)")
print(f"  软件P波到时: {np.nanmin(us_pt_sw):.2f} – {np.nanmax(us_pt_sw):.2f} μs")

# ═══════════════════════════════════════════════════════════════════════════
# § 2  加载对零校准波形 (chushi.csv)
# ═══════════════════════════════════════════════════════════════════════════
print("\n加载校准波形 (chushi.csv)...")
try:
    cal_raw   = pd.read_csv(CAL_FILE, header=None, skiprows=154,
                            low_memory=False, dtype=str,
                            encoding='gbk', on_bad_lines='skip')
    cal_time  = pd.to_numeric(cal_raw.iloc[:, 0], errors='coerce').values
    cal_sig   = pd.to_numeric(cal_raw.iloc[:, 1], errors='coerce').values
    # 去除nan
    valid_cal = ~(np.isnan(cal_time) | np.isnan(cal_sig))
    cal_time  = cal_time[valid_cal]
    cal_sig   = cal_sig[valid_cal]
    print(f"  校准波形: {len(cal_time)} 点  "
          f"  振幅范围: [{cal_sig.min()*1e3:.2f}, {cal_sig.max()*1e3:.2f}] mV")
except Exception as e:
    print(f"  警告: 无法加载校准波形 ({e})，使用经验估计系统延时")
    cal_time, cal_sig = np.array([0.0]), np.array([0.0])

# ═══════════════════════════════════════════════════════════════════════════
# § 3  AIC P波拾取函数
# ═══════════════════════════════════════════════════════════════════════════
def butter_bandpass_sos(fs, low, high, order=4):
    nyq = fs / 2
    sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sos

def apply_bp(signal, sos):
    """零相位带通滤波"""
    try:
        return sosfiltfilt(sos, signal.astype(float))
    except Exception:
        return signal.astype(float)

def aic_pick(waveform, t_us_arr, search_start_us, search_end_us):
    """
    AIC P波初至拾取
    返回: (初至时间μs, AIC曲线, 搜索范围)
    """
    N = len(waveform)
    # 搜索索引
    i0 = max(1, np.searchsorted(t_us_arr, search_start_us))
    i1 = min(N-1, np.searchsorted(t_us_arr, search_end_us))
    if i1 <= i0 + 2:
        return np.nan, None, (i0, i1)

    x = waveform - waveform.mean()
    x2 = x * x
    cumsum = np.cumsum(x2)
    total  = cumsum[-1] + 1e-30

    k = np.arange(i0, i1)
    var1 = cumsum[k-1] / k
    var2 = (total - cumsum[k-1]) / (N - k)
    var1 = np.maximum(var1, 1e-30)
    var2 = np.maximum(var2, 1e-30)

    aic = k * np.log(var1) + (N - k) * np.log(var2)

    # 简单平滑（5点移动均值）以减少噪声极小值
    aic_s = np.convolve(aic, np.ones(5)/5, mode='same')
    idx_min = k[np.argmin(aic_s)]
    return t_us_arr[idx_min], aic_s, (i0, i1)

# ─── 对零波形AIC ──────────────────────────────────────────────────────────
sos_bp = butter_bandpass_sos(FS_HZ, BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER)

if len(cal_time) > 100:
    cal_filt  = apply_bp(cal_sig, sos_bp)
    cal_t_us_arr = cal_time
    # 对零搜索范围：0.5 – 40 μs
    t_cal_aic, _, _ = aic_pick(cal_filt, cal_t_us_arr,
                                search_start_us=0.5, search_end_us=40.0)
    print(f"  AIC校准延时 (对零): {t_cal_aic:.3f} μs")
else:
    t_cal_aic = np.nan
    print("  校准波形数据不足，将用经验方法估计系统延时")

# ═══════════════════════════════════════════════════════════════════════════
# § 4  对所有554次扫描执行AIC拾取
# ═══════════════════════════════════════════════════════════════════════════
print("\n对554次超声波扫描执行AIC P波拾取...")
t_arr = wf_time_us.copy()   # 时间轴 μs

us_pt_aic = np.full(n_sw, np.nan)

for i in range(n_sw):
    wf   = wf_data[:, i].astype(float)
    # 去NaN
    if np.sum(np.isnan(wf)) > n_samp * 0.5:
        continue
    wf[np.isnan(wf)] = 0.0
    # 带通滤波
    wf_f = apply_bp(wf, sos_bp)
    # 以软件值为中心确定搜索窗
    sw_ref  = us_pt_sw[i] if not np.isnan(us_pt_sw[i]) else 30.0
    s_start = max(AIC_GLOBAL_START_US, sw_ref - AIC_SEARCH_OFFSET_US)
    s_end   = sw_ref + AIC_SEARCH_WIDTH_US
    t_aic, _, _ = aic_pick(wf_f, t_arr, s_start, s_end)
    us_pt_aic[i] = t_aic

valid_aic = ~np.isnan(us_pt_aic)
print(f"  成功拾取: {valid_aic.sum()} / {n_sw}")
print(f"  AIC P波到时范围: {np.nanmin(us_pt_aic):.2f} – {np.nanmax(us_pt_aic):.2f} μs")
print(f"  软件值范围:      {np.nanmin(us_pt_sw):.2f} – {np.nanmax(us_pt_sw):.2f} μs")

# ─── 差异统计 ──────────────────────────────────────────────────────────────
diff = us_pt_aic - us_pt_sw
diff_valid = diff[valid_aic & ~np.isnan(us_pt_sw)]
print(f"  AIC - 软件 差值: 均值={np.nanmean(diff_valid):.3f} μs  "
      f"  σ={np.nanstd(diff_valid):.3f} μs  "
      f"  范围=[{np.nanmin(diff_valid):.3f}, {np.nanmax(diff_valid):.3f}] μs")

# ─── 系统延时 & Vp计算 ─────────────────────────────────────────────────────
if not np.isnan(t_cal_aic) and t_cal_aic > 0:
    sys_delay = t_cal_aic
    print(f"  系统延时(对零AIC): {sys_delay:.3f} μs")
else:
    # 经验估计：初期稳定段假设Vp_ref = 4800 m/s
    VPR = 4800.0
    t_prop_ref = H_M / VPR * 1e6
    early_aic = us_pt_aic[valid_aic][:20]
    sys_delay = float(np.nanmedian(early_aic) - t_prop_ref) if len(early_aic) else 0.0
    print(f"  系统延时(经验估计, Vp_ref={VPR:.0f}m/s): {sys_delay:.3f} μs")

# AIC Vp
travel_aic = us_pt_aic - sys_delay
us_Vp_aic  = np.where((travel_aic > 5) & (travel_aic < 200),
                       H_M / (travel_aic * 1e-6) / 1000.0, np.nan)

# 软件 Vp
travel_sw  = us_pt_sw - sys_delay
us_Vp_sw   = np.where((travel_sw > 5) & (travel_sw < 200),
                       H_M / (travel_sw * 1e-6) / 1000.0, np.nan)

print(f"\n  Vp(AIC): {np.nanmin(us_Vp_aic):.2f} – {np.nanmax(us_Vp_aic):.2f} km/s")
print(f"  Vp(软件): {np.nanmin(us_Vp_sw):.2f} – {np.nanmax(us_Vp_sw):.2f} km/s")

# ═══════════════════════════════════════════════════════════════════════════
# § 5  加载声发射数据
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("加载声发射数据...")

def parse_ae_hits(path):
    rows = []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('C:') or line.startswith('Express') \
               or line.startswith('Version') or line.startswith('4/') \
               or line.startswith('128') or line.startswith('ID'):
                continue
            parts = line.split()
            if len(parts) >= 9:
                try:
                    rows.append({'Time': float(parts[1]), 'CH': int(parts[2]),
                                 'RISE': int(parts[3]), 'COUN': int(parts[4]),
                                 'ENER': int(parts[5]), 'DURATION': int(parts[6]),
                                 'AMP': float(parts[7]), 'ABS_E': float(parts[8])})
                except Exception: pass
    return pd.DataFrame(rows)

ae = parse_ae_hits(AE_HITS)
ae = ae[ae['Time'] > 0].sort_values('Time').reset_index(drop=True)
ae_t = ae['Time'].values
print(f"  撞击总数: {len(ae)}   时间: {ae['Time'].min():.2f} – {ae['Time'].max():.2f} s")

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
                    current = {'x':x,'y':y,'z':z,'q':q,'src_amp':sa,'time':None}
                    events.append(current)
                except Exception: current = None
            elif line.startswith('*') and current is not None:
                parts = line.lstrip('*').split()
                if len(parts)>=8:
                    try:
                        t = float(parts[0])
                        if current['time'] is None: current['time'] = t
                    except ValueError: pass
    return pd.DataFrame([e for e in events if e['time'] is not None])

evts = parse_ae_events(AE_EVTS)
print(f"  已定位事件: {len(evts)}")

cum_dfs = []
for i, f in enumerate(AE_CUM_FILES):
    try:
        df = pd.read_csv(f, sep=r'\s+', skiprows=1,
                         names=['Time', f'CH{i+1}'], engine='python')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        cum_dfs.append(df)
    except Exception: pass

# ═══════════════════════════════════════════════════════════════════════════
# § 6  超声波干扰时间窗统计分析（dt分布法）
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("分析超声波干扰持续时间 (dt分布法)...")

# 每个AE撞击 → 距最近的前一次US激发的时间差 dt
dt_arr = np.full(len(ae_t), np.nan)
period_approx = np.median(np.diff(us_ts))  # ≈ 3.13 s

for i, t in enumerate(ae_t):
    prev = us_ts[us_ts <= t]
    if len(prev):
        dt_arr[i] = t - prev[-1]

# dt分布（仅考虑同一脉冲周期内的撞击：dt < period）
dt_valid = dt_arr[(dt_arr >= 0) & (dt_arr < period_approx)]

# 用直方图找到干扰持续时间
BIN_SIZE = 0.05   # 50 ms bins
bins_dt   = np.arange(0, period_approx, BIN_SIZE)
dt_hist, _ = np.histogram(dt_valid, bins=bins_dt)
dt_centers = (bins_dt[:-1] + bins_dt[1:]) / 2

# 背景率（取dt = 1.5*period 到 period段的平均，作为无干扰参考）
bg_mask   = dt_centers > (period_approx * 0.5)
bg_rate   = np.mean(dt_hist[bg_mask]) if bg_mask.any() else 1.0

# 干扰阈值 = 背景率 × 2 → 超过此阈值的连续区域认为是干扰
threshold = bg_rate * 2.0
# 找到干扰结束时间：dt_hist 从超过阈值降回阈值以下
exceed = dt_hist > threshold
if exceed.any():
    # 找最后一个超过阈值的bin
    last_idx   = np.where(exceed)[0][-1]
    window_end = dt_centers[last_idx] + BIN_SIZE
else:
    window_end = 0.5  # 默认500ms

# 确保最小50ms
window_end = max(window_end, 0.05)
print(f"  脉冲平均间隔: {period_approx:.3f} s")
print(f"  背景撞击率: {bg_rate:.1f} hits/bin ({BIN_SIZE*1000:.0f}ms bin)")
print(f"  统计确定的干扰持续时间: {window_end*1000:.0f} ms")

# 应用干扰屏蔽
US_MASK_PRE  = 0.05   # 激发前50ms
US_MASK_POST = window_end

contam = np.zeros(len(ae_t), dtype=bool)
for t_us in us_ts:
    contam |= (ae_t >= t_us - US_MASK_PRE) & (ae_t <= t_us + US_MASK_POST)

ae_clean  = ae[~contam].reset_index(drop=True)
ae_contam = ae[contam].reset_index(drop=True)
n_total   = len(ae)
n_contam  = contam.sum()
n_clean   = n_total - n_contam
pct       = 100. * n_contam / n_total

print(f"  屏蔽窗: 前 {US_MASK_PRE*1000:.0f}ms / 后 {US_MASK_POST*1000:.0f}ms")
print(f"  原始撞击: {n_total}  →  干扰: {n_contam} ({pct:.1f}%)  →  真实: {n_clean} ({100-pct:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# § 7  绘图
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("生成图表...")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图1  对零校准波形 + 几个典型测试波形 + AIC拾取结果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig1 = plt.figure(figsize=(16, 10))
gs1  = gridspec.GridSpec(2, 4, figure=fig1, hspace=0.45, wspace=0.38)
fig1.suptitle('AIC P波拾取示例  |  典型波形（滤波后）与拾取结果',
              fontsize=12, fontweight='bold')

# 对零校准波形
ax_cal = fig1.add_subplot(gs1[0, :2])
if len(cal_time) > 100:
    cal_filt_plot = apply_bp(cal_sig, sos_bp)
    t_mask_cal = cal_time <= 60
    ax_cal.plot(cal_time[t_mask_cal], cal_sig[t_mask_cal]*1000,
                color='gray', lw=0.8, alpha=0.6, label='原始')
    ax_cal.plot(cal_time[t_mask_cal], cal_filt_plot[t_mask_cal]*1000,
                color='steelblue', lw=1.2, label='带通滤波后')
    if not np.isnan(t_cal_aic):
        ax_cal.axvline(t_cal_aic, color='red', lw=1.5, ls='--',
                       label=f'AIC拾取 = {t_cal_aic:.2f} μs')
else:
    ax_cal.text(0.5, 0.5, '校准波形数据不可用', transform=ax_cal.transAxes,
                ha='center', va='center', fontsize=11, color='gray')
ax_cal.set_xlabel('时间 (μs)')
ax_cal.set_ylabel('振幅 (mV)')
ax_cal.set_title('对零校准波形（面-面接触，系统延时标定）')
ax_cal.legend(fontsize=8)

# 选取4个时间点的测试波形展示
test_times  = [us_ts[5], us_ts[len(us_ts)//4], us_ts[len(us_ts)*3//4], us_ts[-20]]
test_labels = ['初期', '1/4段', '3/4段', '临近破坏']
test_colors = ['steelblue', 'green', 'orange', 'red']

for col_idx, (t_show, lbl, clr) in enumerate(zip(test_times, test_labels, test_colors)):
    sw_idx = np.argmin(np.abs(us_ts - t_show))
    ax_t   = fig1.add_subplot(gs1[1, col_idx] if col_idx < 4 else gs1[0, col_idx-2+2])

    wf_raw  = wf_data[:, sw_idx].astype(float)
    wf_raw[np.isnan(wf_raw)] = 0.0
    wf_f    = apply_bp(wf_raw, sos_bp)

    sw_ref  = us_pt_sw[sw_idx] if not np.isnan(us_pt_sw[sw_idx]) else 30.0
    s_start = max(AIC_GLOBAL_START_US, sw_ref - AIC_SEARCH_OFFSET_US)
    s_end   = sw_ref + AIC_SEARCH_WIDTH_US
    t_aic_i, aic_curve, (i0, i1) = aic_pick(wf_f, t_arr, s_start, s_end)

    # 只显示0-120μs
    plot_mask = t_arr <= 120
    ax_t.plot(t_arr[plot_mask], wf_raw[plot_mask]*1e3,
              color='lightgray', lw=0.7, label='原始')
    ax_t.plot(t_arr[plot_mask], wf_f[plot_mask]*1e3,
              color=clr, lw=1.0, alpha=0.9, label='滤波后')

    # AIC曲线（归一化后叠加显示）
    if aic_curve is not None:
        aic_t    = t_arr[i0:i1]
        plot_a   = aic_t <= 120
        aic_norm = (aic_curve[plot_a] - aic_curve[plot_a].min())
        aic_norm = aic_norm / (aic_norm.max() + 1e-30) * np.abs(wf_f[plot_mask]).max()*1e3
        ax_t.plot(aic_t[plot_a], -aic_norm, color='purple', lw=0.8,
                  ls=':', alpha=0.7, label='−AIC(归一)')

    if not np.isnan(t_aic_i):
        ax_t.axvline(t_aic_i, color='red', lw=1.5, ls='--',
                     label=f'AIC={t_aic_i:.2f}μs')
    if not np.isnan(us_pt_sw[sw_idx]):
        ax_t.axvline(us_pt_sw[sw_idx], color='navy', lw=1.2, ls=':',
                     label=f'软件={us_pt_sw[sw_idx]:.2f}μs')

    ax_t.set_xlabel('时间 (μs)')
    ax_t.set_ylabel('振幅 (mV)')
    ax_t.set_title(f'{lbl}  t={t_show:.0f}s')
    ax_t.legend(fontsize=7, ncol=2)
    ax_t.set_xlim(0, 120)

# 顶部右侧：拾取差值直方图
ax_diff = fig1.add_subplot(gs1[0, 2:])
d = diff_valid[np.abs(diff_valid) < 10]
ax_diff.hist(d, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
ax_diff.axvline(np.nanmean(d), color='red', lw=1.5, ls='--',
                label=f'均值 {np.nanmean(d):.2f} μs')
ax_diff.set_xlabel('AIC到时 − 软件到时 (μs)')
ax_diff.set_ylabel('频数')
ax_diff.set_title('拾取差值分布（AIC vs 软件）')
ax_diff.legend()

out1 = os.path.join(BASE, 'v2_01_AIC波形拾取示例.png')
fig1.savefig(out1)
plt.close(fig1)
print(f"图1已保存: {out1}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图2  P波到时与速度 —— AIC vs 软件对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), sharex='col')
fig2.suptitle('超声波P波到时与速度：AIC法 vs GCTS软件自动拾取\n'
              'P-wave Arrival & Velocity: AIC Method vs Software Auto-pick',
              fontsize=12, fontweight='bold')

(ax2a, ax2b), (ax2c, ax2d) = axes2

# P波到时
ax2a.plot(us_ts, us_pt_sw,  color='navy',    lw=0.8, alpha=0.8, label='软件自动拾取')
ax2a.plot(us_ts, us_pt_aic, color='crimson', lw=0.8, alpha=0.8, label='AIC拾取')
ax2a.set_ylabel('P波到时 (μs)')
ax2a.set_title('P波到时（全段）')
ax2a.legend()
ax2a.yaxis.set_minor_locator(AutoMinorLocator())

# 到时差
ax2b.plot(us_ts, diff, color='purple', lw=0.6, alpha=0.7)
ax2b.axhline(0, color='black', lw=0.8, ls='--')
ax2b.axhline(np.nanmean(diff_valid), color='red', lw=1, ls=':',
             label=f'均值 {np.nanmean(diff_valid):.2f} μs')
ax2b.set_ylabel('AIC − 软件 (μs)')
ax2b.set_title('拾取差值')
ax2b.legend()

# Vp
valid_both = ~(np.isnan(us_Vp_aic) | np.isnan(us_Vp_sw))
ax2c.plot(us_ts[valid_both], us_Vp_sw[valid_both],
          color='navy',    lw=0.8, alpha=0.8, label='Vp (软件)')
ax2c.plot(us_ts[valid_both], us_Vp_aic[valid_both],
          color='crimson', lw=0.8, alpha=0.8, label='Vp (AIC)')
ax2c.set_ylabel('P波速度 (km/s)')
ax2c.set_xlabel('时间 (s)')
ax2c.set_title('P波速度演化（全段）')
ax2c.legend()
ax2c.yaxis.set_minor_locator(AutoMinorLocator())

# 检测破坏时刻
vp_base = np.nanmedian(us_Vp_aic[~np.isnan(us_Vp_aic)][:30])
fail_mask = us_Vp_aic < (vp_base * 0.7)  # 速度降至基准70%以下
if np.any(fail_mask):
    t_fail = us_ts[np.where(fail_mask)[0][0]]
    for ax in [ax2a, ax2c]:
        ax.axvline(t_fail, color='red', ls=':', lw=1.5, alpha=0.7)
    ax2c.text(t_fail+15, ax2c.get_ylim()[0]*1.05, f'破坏\nt≈{t_fail:.0f}s',
              color='red', fontsize=8)

# Vp散点对比
ax2d.scatter(us_Vp_sw[valid_both], us_Vp_aic[valid_both],
             s=3, alpha=0.4, color='steelblue')
lim = [min(np.nanmin(us_Vp_sw), np.nanmin(us_Vp_aic))*0.95,
       max(np.nanmax(us_Vp_sw), np.nanmax(us_Vp_aic))*1.05]
ax2d.plot(lim, lim, 'r--', lw=1, label='1:1')
ax2d.set_xlabel('Vp 软件 (km/s)')
ax2d.set_ylabel('Vp AIC (km/s)')
ax2d.set_title('两种方法速度对比散点图')
ax2d.legend()
ax2d.set_xlim(lim); ax2d.set_ylim(lim)

plt.tight_layout()
out2 = os.path.join(BASE, 'v2_02_P波速度AIC对比.png')
fig2.savefig(out2)
plt.close(fig2)
print(f"图2已保存: {out2}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图3  dt分布与干扰窗确定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle('超声波干扰时间窗统计分析\n'
              'AE Hit Time Lag from US Pulse  —  Interference Window Characterization',
              fontsize=12, fontweight='bold')

# dt 直方图（全时段）
axes3[0].bar(dt_centers, dt_hist, width=BIN_SIZE*0.9,
             color='steelblue', alpha=0.7, label='全时段 AE 撞击')
axes3[0].axhline(bg_rate,   color='gray',   lw=1.5, ls='--', label=f'背景率 = {bg_rate:.1f}')
axes3[0].axhline(threshold, color='orange', lw=1.5, ls='--', label=f'阈值 = 2×背景率')
axes3[0].axvline(window_end, color='red', lw=2, ls='-', label=f'干扰窗 = {window_end*1000:.0f}ms')
axes3[0].set_xlabel('距上次US激发时间 dt (s)')
axes3[0].set_ylabel('撞击数 / bin')
axes3[0].set_title('dt 分布直方图（全时段）')
axes3[0].legend(fontsize=8)

# 早期阶段（t<200 s）dt分布（近似"纯干扰"）
early_mask = (ae_t < 200) & (dt_arr >= 0) & (dt_arr < period_approx)
dt_early   = dt_arr[early_mask]
early_hist, _ = np.histogram(dt_early, bins=bins_dt)
axes3[1].bar(dt_centers, early_hist, width=BIN_SIZE*0.9,
             color='tomato', alpha=0.7, label='早期 (t < 200 s)')
axes3[1].axvline(window_end, color='red', lw=2, ls='-', label=f'干扰窗 {window_end*1000:.0f}ms')
axes3[1].set_xlabel('dt (s)')
axes3[1].set_ylabel('撞击数 / bin')
axes3[1].set_title('早期dt分布（加载前期，近似纯干扰）')
axes3[1].legend(fontsize=8)

# 不同窗口大小对应的去除比例
win_range  = np.arange(0.05, 2.5, 0.05)
pct_remove = []
for wp in win_range:
    c = np.zeros(len(ae_t), dtype=bool)
    for t_us in us_ts:
        c |= (ae_t >= t_us - US_MASK_PRE) & (ae_t <= t_us + wp)
    pct_remove.append(100.*c.sum()/n_total)
axes3[2].plot(win_range * 1000, pct_remove, color='steelblue', lw=2)
axes3[2].axvline(window_end * 1000, color='red', lw=2, ls='--',
                 label=f'选定窗口 {window_end*1000:.0f}ms  →  {pct:.1f}%')
axes3[2].set_xlabel('后向屏蔽窗 (ms)')
axes3[2].set_ylabel('去除撞击比例 (%)')
axes3[2].set_title('窗口大小 vs 去除比例')
axes3[2].legend(fontsize=9)
axes3[2].fill_between(win_range*1000, pct_remove, alpha=0.15, color='steelblue')

plt.tight_layout()
out3 = os.path.join(BASE, 'v2_03_干扰时间窗分析.png')
fig3.savefig(out3)
plt.close(fig3)
print(f"图3已保存: {out3}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图4  全段干扰对比图（6通道 × 全时间段）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("生成全段干扰对比图（6通道）...")

T_MAX = ae['Time'].max() + 20
ALPHA_DOT = 0.45
SZ        = 2

fig4 = plt.figure(figsize=(20, 26))
gs4  = gridspec.GridSpec(6, 2, figure=fig4,
                         hspace=0.12, wspace=0.08,
                         top=0.94, bottom=0.05, left=0.07, right=0.97)
fig4.suptitle(
    f'声发射振幅 — 超声波干扰去除 全段对比图 (0 – {T_MAX:.0f} s)\n'
    f'AE Amplitude: Full Segment Before vs. After US Interference Removal\n'
    f'干扰屏蔽窗: 前 {US_MASK_PRE*1000:.0f}ms / 后 {US_MASK_POST*1000:.0f}ms'
    f'   去除 {n_contam} hits ({pct:.1f}%)',
    fontsize=12, fontweight='bold')

amp_ylim = [ae['AMP'].min() - 5, ae['AMP'].max() + 5]

for ch in range(1, 7):
    row = ch - 1

    # 原始数据
    axL = fig4.add_subplot(gs4[row, 0])
    d_all  = ae[ae['CH'] == ch]
    d_ct   = ae_contam[ae_contam['CH'] == ch]
    d_cl   = ae_clean[ae_clean['CH'] == ch]

    # 绘制：先画干扰（灰色），再画真实（彩色）
    if len(d_ct):
        axL.scatter(d_ct['Time'], d_ct['AMP'],
                    s=SZ, alpha=ALPHA_DOT*0.6, color='silver', zorder=1)
    if len(d_cl):
        axL.scatter(d_cl['Time'], d_cl['AMP'],
                    s=SZ, alpha=ALPHA_DOT, color=CH_COLORS[ch], zorder=2)

    axL.set_ylabel(f'CH{ch}\n振幅(dB)', fontsize=9)
    axL.set_ylim(amp_ylim)
    axL.set_xlim(0, T_MAX)
    axL.yaxis.set_major_locator(MultipleLocator(20))
    axL.yaxis.set_minor_locator(MultipleLocator(10))
    if row == 0:
        axL.set_title(f'原始数据  (共 {n_total} hits)', fontsize=11, pad=8)
        # 添加图例
        from matplotlib.lines import Line2D
        leg_h = [Line2D([0],[0], marker='o', ls='None', color='silver',
                         markersize=4, label=f'超声波干扰 ({n_contam})'),
                 Line2D([0],[0], marker='o', ls='None', color=CH_COLORS[ch],
                         markersize=4, label=f'真实AE ({n_clean})')]
        axL.legend(handles=leg_h, fontsize=8, loc='upper left')

    # 清洁数据（右列）
    axR = fig4.add_subplot(gs4[row, 1], sharey=axL)
    if len(d_cl):
        axR.scatter(d_cl['Time'], d_cl['AMP'],
                    s=SZ, alpha=ALPHA_DOT, color=CH_COLORS[ch])
    axR.set_xlim(0, T_MAX)
    axR.set_ylim(amp_ylim)
    axR.yaxis.set_major_locator(MultipleLocator(20))
    if row == 0:
        axR.set_title(f'去超声波干扰后  ({n_clean} hits, {100-pct:.1f}%)',
                      fontsize=11, pad=8)

    # x轴标签只在最后一行显示
    if row < 5:
        axL.tick_params(labelbottom=False)
        axR.tick_params(labelbottom=False)
    else:
        axL.set_xlabel('时间 (s)')
        axR.set_xlabel('时间 (s)')

    # 右轴y标签仅右侧显示
    axR.tick_params(labelleft=False)

    # 破坏时刻标线
    if np.any(fail_mask):
        axL.axvline(t_fail, color='red', lw=0.8, ls=':', alpha=0.6)
        axR.axvline(t_fail, color='red', lw=0.8, ls=':', alpha=0.6)

out4 = os.path.join(BASE, 'v2_04_全段干扰对比图.png')
fig4.savefig(out4)
plt.close(fig4)
print(f"图4已保存: {out4}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图5  综合分析（AIC Vp + AE去干扰后 + 累计撞击）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig5, axes5 = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
fig5.suptitle('超声波与声发射综合分析（AIC法）\n'
              'Combined Analysis: Ultrasonic (AIC) + AE (Interference Removed)',
              fontsize=12, fontweight='bold')

ax5a, ax5b, ax5c, ax5d = axes5

# Vp (AIC)
m_vp = ~np.isnan(us_Vp_aic)
ax5a.plot(us_ts[m_vp], us_Vp_aic[m_vp], color='crimson', lw=1, alpha=0.85, label='Vp (AIC)')
ax5a.fill_between(us_ts[m_vp], us_Vp_aic[m_vp], alpha=0.12, color='crimson')
ax5a.set_ylabel('Vp (km/s)')
ax5a.set_title('P波速度演化 (AIC法)')
ax5a.yaxis.set_minor_locator(AutoMinorLocator())
ax5a.legend(loc='upper left')

# 全通道AE振幅
for ch in range(1, 7):
    d = ae_clean[ae_clean['CH'] == ch]
    ax5b.scatter(d['Time'], d['AMP'], s=1.5, alpha=0.35,
                 color=CH_COLORS[ch], label=f'CH{ch}')
ax5b.set_ylabel('AE振幅 (dB)')
ax5b.set_title('声发射振幅（超声波干扰去除后）')
h = [plt.Line2D([0],[0], marker='o', ls='None', color=CH_COLORS[i+1],
                 markersize=5, label=f'CH{i+1}') for i in range(6)]
ax5b.legend(handles=h, ncol=6, fontsize=8, loc='upper left', framealpha=0.6)

# AE绝对能量（对数）
ax5c.scatter(ae_clean['Time'], ae_clean['ABS_E'], s=1.5, alpha=0.3, color='darkred')
ax5c.set_yscale('log')
ax5c.set_ylabel('绝对能量 (aJ)')
ax5c.set_title('声发射绝对能量（去干扰后）')

# 累计撞击数 对比
ae_cl_s = ae_clean.sort_values('Time')
ae_or_s = ae.sort_values('Time')
ax5d.plot(ae_or_s['Time'], np.arange(1, n_total+1),
          color='tomato', lw=1.2, ls='--', alpha=0.7, label=f'原始 ({n_total})')
ax5d.plot(ae_cl_s['Time'], np.arange(1, n_clean+1),
          color='steelblue', lw=1.5, alpha=0.9, label=f'去干扰后 ({n_clean})')
ax5d.set_ylabel('累计撞击数')
ax5d.set_xlabel('时间 (s)')
ax5d.set_title('累计撞击数（含/不含超声波干扰）')
ax5d.legend()

if np.any(fail_mask):
    for ax in axes5:
        ax.axvline(t_fail, color='red', ls='--', lw=1.2, alpha=0.7)
    axes5[0].text(t_fail+15, ax5a.get_ylim()[0]*1.05,
                  f'破坏\nt≈{t_fail:.0f}s', color='red', fontsize=9)

plt.tight_layout()
out5 = os.path.join(BASE, 'v2_05_综合分析.png')
fig5.savefig(out5)
plt.close(fig5)
print(f"图5已保存: {out5}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 保存AIC拾取结果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df_vp = pd.DataFrame({'time_s': us_ts,
                      'pwave_sw_us': us_pt_sw,
                      'pwave_aic_us': us_pt_aic,
                      'Vp_sw_km': us_Vp_sw,
                      'Vp_aic_km': us_Vp_aic})
out_vp = os.path.join(BASE, 'v2_Vp_AIC.csv')
df_vp.to_csv(out_vp, index=False)

out_ae = os.path.join(BASE, 'v2_AE_clean.csv')
ae_clean.to_csv(out_ae, index=False)

# ─── 汇总统计 ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("统计汇总 | Summary")
print("="*60)
print(f"\n[超声波 P波速度 (AIC)]")
print(f"  系统延时:         {sys_delay:.2f} μs")
print(f"  初始 Vp:          {np.nanmean(us_Vp_aic[:10]):.3f} km/s")
print(f"  峰值 Vp:          {np.nanmax(us_Vp_aic):.3f} km/s")
print(f"  破坏前 Vp:        {np.nanmean(us_Vp_aic[np.where(fail_mask)[0][0]-5:np.where(fail_mask)[0][0]] if fail_mask.any() else us_Vp_aic[-10:]):.3f} km/s")
print(f"  AIC vs 软件 均值差: {np.nanmean(diff_valid):.3f} μs  σ={np.nanstd(diff_valid):.3f} μs")

print(f"\n[声发射干扰去除]")
print(f"  干扰时间窗:       前 {US_MASK_PRE*1000:.0f}ms / 后 {US_MASK_POST*1000:.0f}ms")
print(f"  原始撞击数:       {n_total}")
print(f"  干扰撞击数:       {n_contam} ({pct:.1f}%)")
print(f"  真实AE撞击数:     {n_clean} ({100-pct:.1f}%)")

print(f"\n输出文件:")
for f in [out1, out2, out3, out4, out5, out_vp, out_ae]:
    print(f"  {f}")
print("\n分析完成！")
