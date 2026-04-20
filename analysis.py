#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花岗岩单轴压缩试验 - 超声波与声发射综合分析
Granite Uniaxial Compression Test - Ultrasonic & Acoustic Emission Analysis

样品 / Sample: 花岗岩 Granite  Φ50mm × H100mm
测试日期 / Date: 2026-04-15
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import linregress
import warnings, os

warnings.filterwarnings('ignore')

# ─── 中文字体设置 ────────────────────────────────────────────────────────────
for font in ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams['font.family'] = font
        break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False

# ─── 全局绘图参数 ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# ─── 路径配置 ────────────────────────────────────────────────────────────────
BASE = r'g:\Cursor project\ZCY-shengfashe'
US_FILE  = os.path.join(BASE, '超声波', '04-15 - ultrasonics data.csv')
AE_HITS  = os.path.join(BASE, '声发射', '04-15-hits-振铃计数、能量等.TXT')
AE_EVTS  = os.path.join(BASE, '声发射', '04-15-声发射事件.TXT')
AE_CUM_FILES = [os.path.join(BASE, '声发射', f'04-15-hit-累计撞击数-{i}.TXT') for i in range(1, 7)]
AE_CUM_SUM   = os.path.join(BASE, '声发射', '04-15-hit-累计撞击数汇总-1~6.TXT')

# ─── 试验参数 ────────────────────────────────────────────────────────────────
H_MM  = 100.0                   # 样品高度 mm
H_M   = H_MM / 1000.0           # 样品高度 m
FS_HZ = 40e6                    # 超声波采样率 Hz

# 超声波干扰时间窗（每次激发后屏蔽的声发射时间）
US_MASK_PRE_S  = 0.05           # 激发前 50 ms（时间同步误差裕量）
US_MASK_POST_S = 0.30           # 激发后 300 ms（主波 + 多次反射衰减）

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 加载超声波数据
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("加载超声波数据...")
us_raw = pd.read_csv(US_FILE, header=None, low_memory=False)

# 第3行 (index 2)：每次激发的时间戳 (s)
us_ts   = pd.to_numeric(us_raw.iloc[2, 1:], errors='coerce').dropna().values
# 第6行 (index 5)：GCTS软件已自动拾取的P波到时 (μs)
us_pt   = pd.to_numeric(us_raw.iloc[5, 1:len(us_ts)+1], errors='coerce').values

n_sweeps = min(len(us_ts), len(us_pt))
us_ts = us_ts[:n_sweeps]
us_pt = us_pt[:n_sweeps]

print(f"  超声波扫描次数: {n_sweeps}")
print(f"  时间范围: {us_ts[0]:.1f} – {us_ts[-1]:.1f} s  ({(us_ts[-1]-us_ts[0])/60:.1f} min)")
print(f"  P波到时范围: {np.nanmin(us_pt):.2f} – {np.nanmax(us_pt):.2f} μs")

# ─── 计算P波速度 ──────────────────────────────────────────────────────────────
# 剔除明显异常的P波到时（<10 μs 或 >60 μs）
valid_pt = np.where((us_pt > 10) & (us_pt < 60), us_pt, np.nan)

# 用前20个稳定值估计系统延时
# 初始未扰动花岗岩参考Vp ≈ 4500 m/s（含初始裂缝）
# 系统延时 = 测量到时 - 理论传播时间
VPR_INIT = 4500.0   # m/s  参考速度
t_prop_ref = H_M / VPR_INIT * 1e6   # μs  参考传播时间
early = valid_pt[~np.isnan(valid_pt)][:20]
sys_delay = float(np.nanmedian(early) - t_prop_ref) if len(early) > 0 else 0.0

travel_us = valid_pt - sys_delay          # 净传播时间 μs
us_Vp_ms  = H_M / (travel_us * 1e-6)     # m/s
us_Vp_km  = us_Vp_ms / 1000.0            # km/s
# 剔除不合理值
us_Vp_km  = np.where((us_Vp_km > 1.0) & (us_Vp_km < 10.0), us_Vp_km, np.nan)

print(f"  估算系统延时: {sys_delay:.2f} μs")
print(f"  Vp 范围 (有效): {np.nanmin(us_Vp_km):.2f} – {np.nanmax(us_Vp_km):.2f} km/s")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 加载声发射撞击数据
# ═══════════════════════════════════════════════════════════════════════════════
print("\n加载声发射撞击数据...")

def parse_ae_hits(path):
    """解析MISTRAS Express AE撞击文件"""
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
                    rows.append({
                        'ID':       int(parts[0]),
                        'Time':     float(parts[1]),
                        'CH':       int(parts[2]),
                        'RISE':     int(parts[3]),
                        'COUN':     int(parts[4]),
                        'ENER':     int(parts[5]),
                        'DURATION': int(parts[6]),
                        'AMP':      float(parts[7]),
                        'ABS_E':    float(parts[8]),
                    })
                except (ValueError, IndexError):
                    continue
    return pd.DataFrame(rows)

ae = parse_ae_hits(AE_HITS)
ae = ae[ae['Time'] > 0].sort_values('Time').reset_index(drop=True)
print(f"  总撞击数: {len(ae)}")
print(f"  时间范围: {ae['Time'].min():.2f} – {ae['Time'].max():.2f} s")
print(f"  通道分布:\n    {ae['CH'].value_counts().sort_index().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 加载声发射事件（已定位）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n加载声发射事件数据...")

def parse_ae_events(path):
    """解析含定位坐标的声发射事件文件"""
    events = []
    current_event = None
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('* Gp#'):
                # 解析事件头: * Gp# N[chans] x,y,z = X, Y, Z, q = Q ...
                try:
                    # 坐标
                    xyz_part = line.split('x,y,z =')[1].split(',')
                    x = float(xyz_part[0].strip())
                    y = float(xyz_part[1].strip())
                    z = float(xyz_part[2].split(',')[0].strip())
                    q_str = line.split('q =')[1].strip()
                    q = float(q_str.split()[0])
                    # 震源振幅
                    amp_str = line.split('Src Amplitude =')[1].strip() if 'Src Amplitude' in line else '0'
                    src_amp = float(amp_str.split()[0])
                    current_event = {'x': x, 'y': y, 'z': z, 'q': q, 'src_amp': src_amp,
                                     'time': None, 'hits': []}
                    events.append(current_event)
                except Exception:
                    current_event = None
            elif line.startswith('*') and current_event is not None:
                parts = line.lstrip('*').split()
                if len(parts) >= 8:
                    try:
                        t = float(parts[0])
                        if current_event['time'] is None:
                            current_event['time'] = t
                        current_event['hits'].append(t)
                    except ValueError:
                        pass
    df = pd.DataFrame([{k: v for k, v in ev.items() if k != 'hits'}
                        for ev in events if ev['time'] is not None])
    return df

evts = parse_ae_events(AE_EVTS)
print(f"  总声发射事件: {len(evts)}")
if len(evts):
    print(f"  时间范围: {evts['time'].min():.1f} – {evts['time'].max():.1f} s")
    print(f"  坐标范围: x[{evts['x'].min():.1f},{evts['x'].max():.1f}] "
          f"y[{evts['y'].min():.1f},{evts['y'].max():.1f}] "
          f"z[{evts['z'].min():.1f},{evts['z'].max():.1f}] mm")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 加载累计撞击数
# ═══════════════════════════════════════════════════════════════════════════════
cum_dfs = []
for i, f in enumerate(AE_CUM_FILES):
    try:
        df = pd.read_csv(f, sep=r'\s+', skiprows=1,
                         names=['Time', f'CH{i+1}'], engine='python')
        df['Time']      = pd.to_numeric(df['Time'],      errors='coerce')
        df[f'CH{i+1}'] = pd.to_numeric(df[f'CH{i+1}'], errors='coerce')
        cum_dfs.append(df.dropna())
    except Exception as e:
        print(f"  警告: 无法读取累计文件 {f}: {e}")

try:
    cum_sum = pd.read_csv(AE_CUM_SUM, sep=r'\s+', skiprows=1,
                          names=['Time', 'Total'], engine='python')
    cum_sum['Time']  = pd.to_numeric(cum_sum['Time'],  errors='coerce')
    cum_sum['Total'] = pd.to_numeric(cum_sum['Total'], errors='coerce')
    cum_sum = cum_sum.dropna()
except Exception:
    cum_sum = None

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 超声波干扰去除
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"超声波干扰滤除 (时间窗: -{US_MASK_PRE_S*1000:.0f}ms / +{US_MASK_POST_S*1000:.0f}ms)")

ae_t = ae['Time'].values
contam = np.zeros(len(ae_t), dtype=bool)
for t_us in us_ts:
    lo = t_us - US_MASK_PRE_S
    hi = t_us + US_MASK_POST_S
    contam |= (ae_t >= lo) & (ae_t <= hi)

ae_clean  = ae[~contam].reset_index(drop=True)
ae_contam = ae[contam].reset_index(drop=True)

n_total   = len(ae)
n_contam  = contam.sum()
n_clean   = len(ae_clean)
pct       = 100.0 * n_contam / n_total

print(f"  原始撞击数:          {n_total:>7d}")
print(f"  判定为超声波干扰:     {n_contam:>7d}  ({pct:.1f}%)")
print(f"  保留真实声发射:       {n_clean:>7d}  ({100-pct:.1f}%)")

# ─── 分析不同掩蔽窗对结果的影响 ───────────────────────────────────────────────
windows_post = [0.05, 0.10, 0.20, 0.30, 0.50]
win_stats = []
for wp in windows_post:
    c = np.zeros(len(ae_t), dtype=bool)
    for t_us in us_ts:
        c |= (ae_t >= t_us - US_MASK_PRE_S) & (ae_t <= t_us + wp)
    win_stats.append({'window_ms': wp*1000, 'removed': c.sum(),
                      'pct': 100.*c.sum()/n_total})
print("\n  掩蔽窗敏感性分析:")
for ws in win_stats:
    print(f"    后向窗 {ws['window_ms']:5.0f} ms → 去除 {ws['removed']:6d} hits ({ws['pct']:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. 辅助计算函数
# ═══════════════════════════════════════════════════════════════════════════════

def hit_rate(times, bin_s=30.0):
    """每时间段撞击率 hits/s"""
    if len(times) == 0:
        return np.array([]), np.array([])
    edges = np.arange(times.min(), times.max() + bin_s, bin_s)
    n, e  = np.histogram(times, bins=edges)
    return (e[:-1] + e[1:]) / 2, n / bin_s

def b_value_mle(amp_dB, amp_min=None):
    """最大似然法估计b值（振幅版）"""
    if amp_min is None:
        amp_min = np.percentile(amp_dB, 5)
    a = amp_dB[amp_dB >= amp_min]
    if len(a) < 20:
        return np.nan
    # Gutenberg-Richter: log10(N) = a - b*M, M = AMP_dB/20
    # MLE: b = 20 / (mean(AMP) - (amp_min - dAMP/2)) * log10(e)
    b = 20 * np.log10(np.e) / (np.mean(a) - amp_min + 2.5)
    return b

def moving_b_value(times, amps, window=300, step=100):
    """滑动窗b值"""
    idx = np.argsort(times)
    t_s, a_s = times[idx], amps[idx]
    b_t, b_v = [], []
    for i in range(0, len(t_s)-window, step):
        b = b_value_mle(a_s[i:i+window])
        if not np.isnan(b):
            b_t.append(np.mean(t_s[i:i+window]))
            b_v.append(b)
    return np.array(b_t), np.array(b_v)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. 绘图
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
CH_COLORS = {i+1: COLORS[i] for i in range(6)}

# ────────────────────────────────────────────────────────────────────────────
# 图1: 超声波P波结果
# ────────────────────────────────────────────────────────────────────────────
fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig1.suptitle('超声波P波监测结果 | Ultrasonic P-wave Monitoring\n'
              '花岗岩 Φ50×100mm  单轴压缩', fontsize=12, fontweight='bold')

# 原始P波到时
ax1a.plot(us_ts, valid_pt, color='steelblue', lw=0.8, alpha=0.8, label='P波到时')
ax1a.set_ylabel('P波到时 (μs)')
ax1a.set_title('P波到时随时间变化（GCTS软件自动拾取）')
ax1a.yaxis.set_minor_locator(AutoMinorLocator())
ax1a.legend(loc='upper right')

# 标注破坏区域（P波到时突增）
pt_baseline = np.nanmedian(valid_pt[~np.isnan(valid_pt)][:50])
failure_mask = valid_pt > (pt_baseline + 5.0)
if np.any(failure_mask):
    ax1a.axvspan(us_ts[np.where(failure_mask)[0][0]], us_ts[-1],
                 alpha=0.12, color='red', label='疑似破坏区')
    ax1a.legend(loc='upper right')

# P波速度
ax1b.plot(us_ts, us_Vp_km, color='crimson', lw=0.8, alpha=0.8, label='Vp')
ax1b.fill_between(us_ts, us_Vp_km, alpha=0.1, color='crimson')
ax1b.set_ylabel('P波速度 Vp (km/s)')
ax1b.set_xlabel('时间 (s)')
ax1b.set_title(f'P波速度随时间变化（系统延时估计 {sys_delay:.1f} μs）')
ax1b.yaxis.set_minor_locator(AutoMinorLocator())
ax1b.legend(loc='upper left')

# 添加速度变化率趋势注释
try:
    vp_valid = us_Vp_km[~np.isnan(us_Vp_km)]
    ts_valid = us_ts[~np.isnan(us_Vp_km)]
    ax1b.annotate(f'初始Vp≈{vp_valid[:10].mean():.2f} km/s',
                  xy=(ts_valid[5], vp_valid[5]),
                  xytext=(ts_valid[5]+100, vp_valid[5]+0.3),
                  arrowprops=dict(arrowstyle='->', color='gray'),
                  fontsize=9)
except Exception:
    pass

plt.tight_layout()
out1 = os.path.join(BASE, '01_超声波P波结果.png')
fig1.savefig(out1)
plt.close(fig1)
print(f"\n图1已保存: {out1}")

# ────────────────────────────────────────────────────────────────────────────
# 图2: 声发射原始数据 vs 去干扰后对比
# ────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
fig2.suptitle('声发射数据 — 超声波干扰去除对比\n'
              'AE Data: Before vs. After Ultrasonic Interference Removal',
              fontsize=12, fontweight='bold')

ax_titles_left  = ['振幅-时间（原始）', '振铃计数-时间（原始）', '绝对能量-时间（原始）']
ax_titles_right = [f'振幅-时间（去干扰后，保留{100-pct:.0f}%）',
                   '振铃计数-时间（去干扰后）',
                   '绝对能量-时间（去干扰后）']

datasets = [(ae, '原始'), (ae_clean, '去干扰')]
cols_y   = ['AMP', 'COUN', 'ABS_E']
ylabels  = ['振幅 (dB)', '振铃计数', '绝对能量 (aJ)']

for row, (col_y, ylabel) in enumerate(zip(cols_y, ylabels)):
    for col, (ds, label) in enumerate(datasets):
        ax = axes2[row, col]
        for ch in range(1, 7):
            d = ds[ds['CH'] == ch]
            if len(d) == 0:
                continue
            ax.scatter(d['Time'], d[col_y],
                       s=1, alpha=0.35, color=CH_COLORS[ch], label=f'CH{ch}')
        ax.set_ylabel(ylabel)
        title = ax_titles_left[row] if col == 0 else ax_titles_right[row]
        ax.set_title(title)
        if col_y == 'ABS_E':
            ax.set_yscale('log')
        if row == 2:
            ax.set_xlabel('时间 (s)')
        if row == 0:
            handles = [plt.Line2D([0],[0], marker='o', ls='None',
                                   color=CH_COLORS[i+1], markersize=5, label=f'CH{i+1}')
                       for i in range(6)]
            ax.legend(handles=handles, ncol=3, fontsize=8,
                      loc='upper left', framealpha=0.6)

plt.tight_layout()
out2 = os.path.join(BASE, '02_声发射干扰对比.png')
fig2.savefig(out2)
plt.close(fig2)
print(f"图2已保存: {out2}")

# ────────────────────────────────────────────────────────────────────────────
# 图3: 去干扰后声发射深度分析
# ────────────────────────────────────────────────────────────────────────────
fig3 = plt.figure(figsize=(16, 16))
gs3  = gridspec.GridSpec(4, 2, figure=fig3, hspace=0.42, wspace=0.32)
fig3.suptitle('声发射深度分析（超声波干扰去除后）\nAE Analysis After Interference Removal',
              fontsize=12, fontweight='bold')

# 3a: 累计撞击数（各通道）
ax3a = fig3.add_subplot(gs3[0, :])
for i, cdf in enumerate(cum_dfs):
    ax3a.plot(cdf['Time'], cdf[f'CH{i+1}'],
              color=COLORS[i], lw=1.5, label=f'CH{i+1}')
if cum_sum is not None and len(cum_sum):
    ax3a.plot(cum_sum['Time'], cum_sum['Total'],
              'k--', lw=2, label='汇总')
ax3a.set_ylabel('累计撞击数')
ax3a.set_title('各通道累计撞击数（原始数据，仅供参考）')
ax3a.legend(ncol=4, fontsize=9)

# 3b: 撞击率对比（原始 vs 去干扰）
ax3b = fig3.add_subplot(gs3[1, 0])
t_r,  hr_r = hit_rate(ae['Time'].values,       bin_s=30)
t_c,  hr_c = hit_rate(ae_clean['Time'].values, bin_s=30)
if len(t_r):
    ax3b.plot(t_r, hr_r, color='tomato',    lw=1.5, alpha=0.8, label='原始')
if len(t_c):
    ax3b.plot(t_c, hr_c, color='steelblue', lw=1.5, alpha=0.9, label='去干扰后')
ax3b.set_ylabel('撞击率 (hits/s)')
ax3b.set_xlabel('时间 (s)')
ax3b.set_title('撞击率对比（30s窗）')
ax3b.legend()

# 3c: 累计AE能量（去干扰后）
ax3c = fig3.add_subplot(gs3[1, 1])
ae_cs = ae_clean.sort_values('Time')
ax3c.plot(ae_cs['Time'], ae_cs['ABS_E'].cumsum(),
          color='darkgreen', lw=2)
ax3c.set_ylabel('累计绝对能量 (aJ)')
ax3c.set_xlabel('时间 (s)')
ax3c.set_title('累计绝对能量（去干扰后）')

# 3d: 振幅-频度分布与b值
ax3d = fig3.add_subplot(gs3[2, 0])
amp_all = ae_clean['AMP'].dropna().values
if len(amp_all) > 10:
    amp_range = np.arange(amp_all.min(), amp_all.max() + 1, 3)
    cum_n = [np.sum(amp_all >= a) for a in amp_range]
    ax3d.semilogy(amp_range, cum_n, 'ko', ms=3, alpha=0.7, label='实测')
    # 拟合b值
    valid = np.array(cum_n) > 0
    try:
        sl, ic, r, _, _ = linregress(amp_range[valid], np.log10(np.array(cum_n)[valid]))
        b_global = -sl * 20
        ax3d.semilogy(amp_range[valid],
                      10**(ic + sl * amp_range[valid]),
                      'r--', lw=2, label=f'拟合 b = {b_global:.2f}  (R²={r**2:.3f})')
    except Exception:
        b_global = np.nan
    ax3d.set_xlabel('振幅阈值 (dB)')
    ax3d.set_ylabel('累计事件数')
    ax3d.set_title(f'振幅-频度分布（b值 = {b_global:.2f}）')
    ax3d.legend()

# 3e: 滑动窗b值演化
ax3e = fig3.add_subplot(gs3[2, 1])
bv_t, bv_v = moving_b_value(ae_clean['Time'].values,
                             ae_clean['AMP'].values,
                             window=500, step=200)
if len(bv_t):
    ax3e.plot(bv_t, bv_v, 'purple', lw=1.5, alpha=0.9)
    ax3e.axhline(1.5, ls='--', color='gray', lw=0.8, label='参考线 b=1.5')
    ax3e.set_xlabel('时间 (s)')
    ax3e.set_ylabel('b 值')
    ax3e.set_title('滑动窗b值演化（窗=500 hits, 步=200 hits）')
    ax3e.legend()

# 3f: 声发射事件三维定位
if len(evts) > 5:
    ax3f = fig3.add_subplot(gs3[3, 0])
    sc3f = ax3f.scatter(evts['x'], evts['y'],
                        c=evts['time'], cmap='plasma', s=10, alpha=0.6)
    plt.colorbar(sc3f, ax=ax3f, label='时间 (s)')
    ax3f.set_xlabel('x (mm)')
    ax3f.set_ylabel('y (mm)')
    ax3f.set_title(f'声发射事件XY平面定位 (N={len(evts)})')

    ax3g = fig3.add_subplot(gs3[3, 1])
    sc3g = ax3g.scatter(evts['time'], evts['z'],
                        c=evts['src_amp'], cmap='hot', s=10, alpha=0.6,
                        vmin=40, vmax=90)
    plt.colorbar(sc3g, ax=ax3g, label='震源振幅 (dB)')
    ax3g.set_xlabel('时间 (s)')
    ax3g.set_ylabel('z (mm)')
    ax3g.set_title('震源深度 z 随时间变化')

plt.tight_layout()
out3 = os.path.join(BASE, '03_声发射深度分析.png')
fig3.savefig(out3)
plt.close(fig3)
print(f"图3已保存: {out3}")

# ────────────────────────────────────────────────────────────────────────────
# 图4: 超声波 + 声发射综合对比图
# ────────────────────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(5, 1, figsize=(16, 20), sharex=True)
fig4.suptitle('超声波与声发射综合分析 | Combined Ultrasonic & AE Analysis\n'
              '花岗岩 Φ50×100mm  单轴压缩', fontsize=13, fontweight='bold')

(ax4a, ax4b, ax4c, ax4d, ax4e) = axes4

# ── P波速度
ax4a.plot(us_ts, us_Vp_km, color='steelblue', lw=1, alpha=0.85)
ax4a.fill_between(us_ts, us_Vp_km, alpha=0.12, color='steelblue')
ax4a.set_ylabel('Vp (km/s)')
ax4a.set_title('P波速度演化')
ax4a.yaxis.set_minor_locator(AutoMinorLocator())

# ── 超声波P波到时（原始μs）
ax4b.plot(us_ts, valid_pt, color='navy', lw=0.8, alpha=0.7)
ax4b.set_ylabel('P波到时 (μs)')
ax4b.set_title('P波到时（已由软件拾取）')

# ── AE振幅散点（去干扰后）
for ch in range(1, 7):
    d = ae_clean[ae_clean['CH'] == ch]
    ax4c.scatter(d['Time'], d['AMP'],
                 s=1.5, alpha=0.4, color=CH_COLORS[ch], label=f'CH{ch}')
ax4c.set_ylabel('AE振幅 (dB)')
ax4c.set_title('声发射振幅（超声波干扰去除后）')
handles = [plt.Line2D([0],[0], marker='o', ls='None',
                       color=CH_COLORS[i+1], markersize=5, label=f'CH{i+1}')
           for i in range(6)]
ax4c.legend(handles=handles, ncol=6, fontsize=8, loc='upper left', framealpha=0.6)

# ── AE绝对能量（去干扰后，对数坐标）
ax4d.scatter(ae_clean['Time'], ae_clean['ABS_E'],
             s=1.5, alpha=0.3, color='darkred')
ax4d.set_yscale('log')
ax4d.set_ylabel('绝对能量 (aJ)')
ax4d.set_title('声发射绝对能量（去干扰后）')

# ── 累计撞击数对比
if len(ae_clean):
    aec_s = ae_clean.sort_values('Time')
    ax4e.plot(aec_s['Time'], np.arange(1, len(aec_s)+1),
              color='steelblue', lw=1.5, label=f'去干扰 ({n_clean})', alpha=0.9)
ae_s = ae.sort_values('Time')
ax4e.plot(ae_s['Time'], np.arange(1, len(ae_s)+1),
          color='tomato', lw=1.2, ls='--', label=f'原始 ({n_total})', alpha=0.7)
ax4e.set_ylabel('累计撞击数')
ax4e.set_xlabel('时间 (s)')
ax4e.set_title('累计撞击数对比（含/不含超声波干扰）')
ax4e.legend()

# ── 标注破坏时刻（P波到时突增）
if np.any(failure_mask):
    t_fail = us_ts[np.where(failure_mask)[0][0]]
    for ax in axes4:
        ax.axvline(t_fail, color='red', ls='--', lw=1.2, alpha=0.7)
    axes4[0].text(t_fail + 10, ax4a.get_ylim()[0]*1.02,
                  f'破坏\nt={t_fail:.0f}s', color='red', fontsize=8)

plt.tight_layout()
out4 = os.path.join(BASE, '04_综合分析.png')
fig4.savefig(out4)
plt.close(fig4)
print(f"图4已保存: {out4}")

# ────────────────────────────────────────────────────────────────────────────
# 图5: 超声波干扰识别详图（局部展示）
# ────────────────────────────────────────────────────────────────────────────
# 选取一段典型区间展示干扰效果
T_SHOW_START = 100.0
T_SHOW_END   = 160.0
ae_zoom   = ae[(ae['Time'] >= T_SHOW_START) & (ae['Time'] <= T_SHOW_END)]
us_zoom   = us_ts[(us_ts >= T_SHOW_START) & (us_ts <= T_SHOW_END)]
ae_cz     = ae_clean[(ae_clean['Time'] >= T_SHOW_START) & (ae_clean['Time'] <= T_SHOW_END)]

fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig5.suptitle(f'超声波干扰识别与去除 — 局部展示 ({T_SHOW_START:.0f}–{T_SHOW_END:.0f} s)\n'
              'Ultrasonic Interference Identification & Removal (Zoom)',
              fontsize=12, fontweight='bold')

# 原始
for ch in range(1, 7):
    d = ae_zoom[ae_zoom['CH'] == ch]
    ax5a.scatter(d['Time'], d['AMP'], s=4, alpha=0.6, color=CH_COLORS[ch], label=f'CH{ch}')
for t_us in us_zoom:
    ax5a.axvspan(t_us - US_MASK_PRE_S, t_us + US_MASK_POST_S,
                 alpha=0.15, color='orange', zorder=0)
    ax5a.axvline(t_us, color='orange', lw=0.5, alpha=0.5)
ax5a.set_ylabel('振幅 (dB)')
ax5a.set_title(f'原始数据（橙色区域为超声波激发时间窗，去除 {n_contam} 次干扰事件）')
ax5a.legend(ncol=6, fontsize=8, loc='upper right')

# 去干扰后
for ch in range(1, 7):
    d = ae_cz[ae_cz['CH'] == ch]
    ax5b.scatter(d['Time'], d['AMP'], s=4, alpha=0.7, color=CH_COLORS[ch], label=f'CH{ch}')
for t_us in us_zoom:
    ax5b.axvline(t_us, color='orange', lw=0.5, alpha=0.4)
ax5b.set_ylabel('振幅 (dB)')
ax5b.set_xlabel('时间 (s)')
ax5b.set_title('去除超声波干扰后（保留真实声发射信号）')
ax5b.legend(ncol=6, fontsize=8, loc='upper right')

plt.tight_layout()
out5 = os.path.join(BASE, '05_干扰识别局部展示.png')
fig5.savefig(out5)
plt.close(fig5)
print(f"图5已保存: {out5}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. 保存过滤后的声发射数据
# ═══════════════════════════════════════════════════════════════════════════════
out_csv = os.path.join(BASE, 'AE_clean_filtered.csv')
ae_clean.to_csv(out_csv, index=False)
print(f"\n去干扰AE数据已保存: {out_csv}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. 统计摘要
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("统计摘要 | Summary Statistics")
print("=" * 60)
print(f"\n[超声波]")
print(f"  测试时长:          {(us_ts[-1]-us_ts[0])/60:.1f} min ({us_ts[-1]-us_ts[0]:.0f} s)")
print(f"  超声波激发次数:    {n_sweeps}")
print(f"  平均激发间隔:      {np.diff(us_ts).mean():.2f} s")
vp_init  = np.nanmean(us_Vp_km[:20]) if np.any(~np.isnan(us_Vp_km[:20])) else np.nan
vp_peak  = np.nanmax(us_Vp_km)
vp_final = np.nanmean(us_Vp_km[-5:]) if np.any(~np.isnan(us_Vp_km[-5:])) else np.nan
print(f"  初始 Vp:           {vp_init:.3f} km/s")
print(f"  峰值 Vp:           {vp_peak:.3f} km/s")
print(f"  末期 Vp:           {vp_final:.3f} km/s")
pct_change = 100*(vp_final-vp_init)/vp_init if vp_init > 0 else np.nan
print(f"  Vp 总变化量:       {pct_change:.1f}%")

print(f"\n[声发射]")
print(f"  原始撞击总数:      {n_total}")
print(f"  超声波干扰撞击:    {n_contam}  ({pct:.1f}%)")
print(f"  真实AE撞击:        {n_clean}  ({100-pct:.1f}%)")
print(f"  AE事件（已定位）:  {len(evts)}")
if len(ae_clean):
    print(f"  振幅范围:          {ae_clean['AMP'].min():.0f} – {ae_clean['AMP'].max():.0f} dB")
    print(f"  中值振幅:          {ae_clean['AMP'].median():.1f} dB")
    print(f"  总绝对能量:        {ae_clean['ABS_E'].sum():.3e} aJ")
    if 'b_global' in dir() and not np.isnan(b_global):
        print(f"  全局 b 值:         {b_global:.3f}")

print("\n输出文件:")
for f in [out1, out2, out3, out4, out5, out_csv]:
    print(f"  {f}")
print("\n分析完成！")
