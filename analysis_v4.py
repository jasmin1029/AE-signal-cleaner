#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
花岗岩单轴压缩试验 - 改进版综合分析 v4
核心改进：用 2D 密度图 (dt_from_US_pulse, amplitude) 识别干扰
  - 真实 AE：随机，每个 (dt, amp) 格子里 hit 率低
  - 超声波激发干扰：周期性，对应格子 hit 率接近 1 次/cycle
  - 阈值 = 背景率 × DENSITY_FACTOR（数据自适应，不依赖固定振幅带）
  - 先做基础时间窗（0~500ms）保底，再用密度法扩展捡漏，最后保留孤立真实 AE

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
from matplotlib.colors import LogNorm
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
H_MM  = 100.0
H_M   = H_MM / 1000.0
FS_HZ = 40e6
BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER = 50e3, 700e3, 4
AIC_SEARCH_OFFSET_US = 12.0
AIC_SEARCH_WIDTH_US  = 30.0
AIC_GLOBAL_START_US  = 5.0

# 干扰识别参数
US_MASK_PRE   = 0.05   # 基础时间窗：激发前 50 ms
US_MASK_POST  = 0.50   # 基础时间窗：激发后 500 ms
US_EXT_MAX    = 2.50   # 密度法最大扩展窗：激发后 2500 ms

# 2D 密度图参数
DT_BIN_S      = 0.05   # dt 分辨率：50 ms
AMP_BIN_DB    = 2.0    # 振幅分辨率：2 dB
DENSITY_FACTOR = 5.0   # 阈值 = 背景率 × DENSITY_FACTOR
MIN_HITS_PER_CYCLE = 0.05  # 最低阈值：平均每个 cycle 出现 0.05 次

COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
CH_COLORS = {i+1: COLORS[i] for i in range(6)}


# ═══════════════════════════════════════════════════════════════════════════
# § 0  辅助函数
# ═══════════════════════════════════════════════════════════════════════════
def butter_bp_sos(fs, low, high, order=4):
    nyq = fs / 2
    return butter(order, [low/nyq, high/nyq], btype='band', output='sos')

def apply_bp(sig, sos):
    try:
        return sosfiltfilt(sos, sig.astype(float))
    except Exception:
        return sig.astype(float)

def aic_pick(wf, t_us, s0, s1):
    N = len(wf)
    i0 = max(1, int(np.searchsorted(t_us, s0)))
    i1 = min(N-1, int(np.searchsorted(t_us, s1)))
    if i1 <= i0+2:
        return np.nan, None, (i0, i1)
    x = wf - wf.mean()
    cs = np.cumsum(x*x)
    tot = cs[-1] + 1e-30
    k   = np.arange(i0, i1)
    v1  = np.maximum(cs[k-1] / k, 1e-30)
    v2  = np.maximum((tot - cs[k-1]) / (N-k), 1e-30)
    aic = k*np.log(v1) + (N-k)*np.log(v2)
    aic_s = np.convolve(aic, np.ones(5)/5, mode='same')
    return t_us[k[np.argmin(aic_s)]], aic_s, (i0, i1)


# ═══════════════════════════════════════════════════════════════════════════
# § 1  超声波数据
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("加载超声波测试数据...")
us_raw   = pd.read_csv(US_FILE, header=None, low_memory=False, dtype=str)
us_ts    = pd.to_numeric(us_raw.iloc[2, 1:], errors='coerce').dropna().values
n_sw     = len(us_ts)
us_pt_sw = pd.to_numeric(us_raw.iloc[5, 1:n_sw+1], errors='coerce').values

print("  解析波形数据...")
wf_block   = us_raw.iloc[7:, :]
wf_time_us = pd.to_numeric(wf_block.iloc[:, 0], errors='coerce').values
wf_data    = wf_block.iloc[:, 1:n_sw+1].apply(pd.to_numeric, errors='coerce').values
del us_raw, wf_block; gc.collect()

n_samp = wf_data.shape[0]
print(f"  扫描次数: {n_sw}  样点: {n_samp}  时间: {us_ts[0]:.1f}-{us_ts[-1]:.1f} s")

# ═══════════════════════════════════════════════════════════════════════════
# § 2  对零校准 -> 系统延时
# ═══════════════════════════════════════════════════════════════════════════
print("\n加载对零校准波形...")
t_cal_aic = np.nan
sos_bp = butter_bp_sos(FS_HZ, BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER)

try:
    cal_raw  = pd.read_csv(CAL_FILE, header=None, skiprows=154,
                           low_memory=False, dtype=str, encoding='latin-1')
    cal_time = pd.to_numeric(cal_raw.iloc[:, 0], errors='coerce').values
    cal_sig  = pd.to_numeric(cal_raw.iloc[:, 1], errors='coerce').values
    ok = ~(np.isnan(cal_time) | np.isnan(cal_sig))
    cal_time, cal_sig = cal_time[ok], cal_sig[ok]
    cal_filt = apply_bp(cal_sig, sos_bp)
    snr = np.max(np.abs(cal_sig)) / (np.std(cal_sig[:100]) + 1e-30)
    print(f"  {len(cal_time)} pts  SNR={snr:.0f}")
    if snr > 10:
        t_cal_aic, _, _ = aic_pick(cal_filt, cal_time, 0.5, 40.0)
        print(f"  AIC t_cal = {t_cal_aic:.3f} us")
except Exception as e:
    print(f"  警告: {e}")
    cal_time = cal_sig = cal_filt = np.array([0.0])

if np.isnan(t_cal_aic):
    early_sw = us_pt_sw[~np.isnan(us_pt_sw)][:20]
    t_cal_aic = float(np.nanmedian(early_sw) - H_M/4800.0*1e6) if len(early_sw) else 0.0
    print(f"  经验延时 = {t_cal_aic:.3f} us")

sys_delay = t_cal_aic
print(f"  系统延时 = {sys_delay:.3f} us")

# ═══════════════════════════════════════════════════════════════════════════
# § 3  AIC 拾取 -> Vp
# ═══════════════════════════════════════════════════════════════════════════
print("\nAIC 拾取 (554 sweeps)...")
t_arr = wf_time_us.copy()
us_pt_aic = np.full(n_sw, np.nan)
for i in range(n_sw):
    wf = wf_data[:, i].astype(float)
    if np.sum(np.isnan(wf)) > n_samp*0.5:
        continue
    wf[np.isnan(wf)] = 0.0
    wf_f = apply_bp(wf, sos_bp)
    ref  = us_pt_sw[i] if not np.isnan(us_pt_sw[i]) else 30.0
    us_pt_aic[i], _, _ = aic_pick(wf_f, t_arr,
                                   max(AIC_GLOBAL_START_US, ref - AIC_SEARCH_OFFSET_US),
                                   ref + AIC_SEARCH_WIDTH_US)

valid_aic = ~np.isnan(us_pt_aic)
print(f"  成功: {valid_aic.sum()}/{n_sw}")

travel_aic = (us_pt_aic - sys_delay) * 1e-6
us_Vp_aic  = np.where((travel_aic > 5e-6) & (travel_aic < 200e-6),
                       H_M / travel_aic / 1000.0, np.nan)
travel_sw  = (us_pt_sw - sys_delay) * 1e-6
us_Vp_sw   = np.where((travel_sw  > 5e-6) & (travel_sw  < 200e-6),
                       H_M / travel_sw  / 1000.0, np.nan)

diff_valid = (us_pt_aic - us_pt_sw)[valid_aic & ~np.isnan(us_pt_sw)]
print(f"  Vp(AIC): {np.nanmin(us_Vp_aic):.2f}-{np.nanmax(us_Vp_aic):.2f} km/s")

# 破坏时刻
vp_base   = np.nanmedian(us_Vp_aic[valid_aic][:30]) if valid_aic.sum() >= 30 else np.nan
fail_mask = (~np.isnan(us_Vp_aic)) & (us_Vp_aic < vp_base*0.7) if not np.isnan(vp_base) else np.zeros(n_sw, bool)
t_fail    = float(us_ts[np.where(fail_mask)[0][0]]) if np.any(fail_mask) else np.nan

# ═══════════════════════════════════════════════════════════════════════════
# § 4  声发射数据
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
                except Exception:
                    pass
    return pd.DataFrame(rows)

ae     = parse_ae_hits(AE_HITS)
ae     = ae[ae['Time'] > 0].sort_values('Time').reset_index(drop=True)
ae_t   = ae['Time'].values
ae_amp = ae['AMP'].values
ae_dur = ae['DURATION'].values
ae_ris = ae['RISE'].values
ae_ch  = ae['CH'].values
n_hits = len(ae)
print(f"  撞击: {n_hits}  时间: {ae_t.min():.1f}-{ae_t.max():.1f} s")

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
                    q = float(line.split('q =')[1].strip().split()[0])
                    cur = {'x':x,'y':y,'z':z,'q':q,'time':None}
                    events.append(cur)
                except Exception:
                    cur = None
            elif line.startswith('*') and cur is not None:
                p = line.lstrip('*').split()
                if len(p) >= 8:
                    try:
                        t = float(p[0])
                        if cur['time'] is None: cur['time'] = t
                    except ValueError: pass
    return pd.DataFrame([e for e in events if e['time'] is not None])

evts = parse_ae_events(AE_EVTS)
print(f"  事件: {len(evts)}")

# ═══════════════════════════════════════════════════════════════════════════
# § 5  分层干扰识别
#
#  Layer-1  基础时间窗：US激发前50ms / 后500ms 内全部标记
#  Layer-2  2D密度检验：剩余撞击中（dt ∈ [0, US_EXT_MAX]），
#           用 (dt_from_US, amplitude) 二维密度图检测干扰热点。
#           热点判定：cell 内每 cycle 命中率 > 背景率 * DENSITY_FACTOR
#  两层取并集即为最终干扰集
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("分层干扰识别 (Layer-1 时间窗 + Layer-2 2D密度)...")

period_approx = float(np.median(np.diff(us_ts)))
n_cycles      = len(us_ts)

# ── 为每个 hit 计算 dt（距上一次 US 激发的时间差）────────────────────────
# 向量化：先排序 us_ts（已是顺序），用 searchsorted
us_ts_sorted = np.sort(us_ts)
# idx[i] = 在 us_ts_sorted 中 < ae_t[i] 的最右位置
_idx = np.searchsorted(us_ts_sorted, ae_t, side='right') - 1
dt_from_us = np.where(_idx >= 0, ae_t - us_ts_sorted[_idx], np.nan)

# ── Layer-1 基础时间窗 ───────────────────────────────────────────────────
# dt ∈ [-US_MASK_PRE, US_MASK_POST]
# 注意 dt < 0 代表 hit 在 US 激发之前（最多 PRE）
contam_L1 = (dt_from_us >= -US_MASK_PRE) & (dt_from_us <= US_MASK_POST)
# 也要处理：hit 在下一个 US 之前 0~PRE 的情况（即距下一个 US < PRE）
_idx_next = np.searchsorted(us_ts_sorted, ae_t, side='left')
dt_to_next = np.where(_idx_next < n_cycles,
                       us_ts_sorted[np.minimum(_idx_next, n_cycles-1)] - ae_t,
                       np.inf)
contam_L1 |= (dt_to_next >= 0) & (dt_to_next <= US_MASK_PRE)

n_L1 = int(contam_L1.sum())
print(f"  Layer-1 基础时间窗: {n_L1} hits ({100.*n_L1/n_hits:.1f}%)")

# ── Layer-2：仅对 Layer-1 未捕获、但在扩展窗内的 hits 做密度检验 ──────────
in_ext = (~contam_L1) & (dt_from_us >= 0) & (dt_from_us <= US_EXT_MAX)

# 2D 直方图：(dt, amplitude)
dt_edges  = np.arange(0, US_EXT_MAX + DT_BIN_S, DT_BIN_S)
amp_edges = np.arange(35, 109, AMP_BIN_DB)

H2d, _, _ = np.histogram2d(
    dt_from_us[in_ext], ae_amp[in_ext],
    bins=[dt_edges, amp_edges]
)
H_per_cyc = H2d / n_cycles   # 每 cell 每 cycle 的平均命中次数

# 背景率：用 dt ∈ [US_MASK_POST, US_EXT_MAX] 区间估计
bg_start_bin = int(np.searchsorted(dt_edges, US_MASK_POST))
if H_per_cyc[bg_start_bin:, :].size > 0:
    bg_rate = float(np.mean(H_per_cyc[bg_start_bin:, :]))
else:
    bg_rate = 1e-4
bg_rate = max(bg_rate, 1e-5)   # 防止除零

density_thresh = max(bg_rate * DENSITY_FACTOR, MIN_HITS_PER_CYCLE)
hot_cell = H_per_cyc > density_thresh

print(f"  2D密度背景率: {bg_rate:.4f} hits/cell/cycle")
print(f"  密度阈值: {density_thresh:.4f} hits/cell/cycle")
print(f"  热点格子数: {hot_cell.sum()} / {hot_cell.size}")

# 对每个 in_ext 的 hit，查询其 (dt, amp) 格子是否为热点
dt_cidx  = np.clip(np.searchsorted(dt_edges[1:],  dt_from_us), 0, len(dt_edges)-2)
amp_cidx = np.clip(np.searchsorted(amp_edges[1:], ae_amp),     0, len(amp_edges)-2)

contam_L2 = in_ext & hot_cell[dt_cidx, amp_cidx]
n_L2 = int(contam_L2.sum())
print(f"  Layer-2 密度热点: {n_L2} hits ({100.*n_L2/n_hits:.1f}%)")

# ── 合并 & 统计 ───────────────────────────────────────────────────────────
contam   = contam_L1 | contam_L2
n_contam = int(contam.sum())
n_clean  = n_hits - n_contam
pct      = 100.0 * n_contam / n_hits

ae_clean  = ae[~contam].reset_index(drop=True)
ae_contam = ae[contam].reset_index(drop=True)

print(f"  原始: {n_hits}  干扰: {n_contam} ({pct:.1f}%)  真实AE: {n_clean} ({100-pct:.1f}%)")

# 和 v3 对比（v3 去除了 58.2%）
v3_pct = 58.2
print(f"  对比 v3: v3去除{v3_pct:.1f}%  本版去除{pct:.1f}%  "
      f"净回收真实AE {'多' if pct < v3_pct else '少'}{abs(pct-v3_pct):.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# § 6  绘图
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("生成图表...")

def vline_fail(axes_list, t_f):
    if not np.isnan(t_f):
        for ax in axes_list:
            ax.axvline(t_f, color='red', ls='--', lw=1.2, alpha=0.7)

# ─── 干扰振幅频带（用于图中参考线）─────────────────────────────────────────
# 从 L1 早期撞击中读出峰值振幅（仅供可视化用）
_early_L1 = ae_amp[contam_L1 & (ae_t < 200)]
if len(_early_L1) >= 20:
    _h, _e = np.histogram(_early_L1, bins=np.arange(35, 105, 2))
    _c = (_e[:-1] + _e[1:]) / 2
    amp_ref = float(_c[np.argmax(_h)])
else:
    amp_ref = 60.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图1  2D 密度图（核心诊断图）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig1, axes1 = plt.subplots(1, 3, figsize=(20, 7))
fig1.suptitle(
    '2D密度图 (dt_from_US_pulse, Amplitude)  —  干扰热点识别\n'
    '蓝色格子 = 背景（真实AE）  红框 = 密度热点（超声波干扰）',
    fontsize=12, fontweight='bold')

dt_ctrs  = (dt_edges[:-1]  + dt_edges[1:])  / 2
amp_ctrs = (amp_edges[:-1] + amp_edges[1:]) / 2

# 全通道 2D 密度（对数色标）
ax = axes1[0]
Hplot = H_per_cyc.copy(); Hplot[Hplot == 0] = np.nan
im = ax.pcolormesh(dt_edges, amp_edges, Hplot.T, shading='flat',
                   norm=LogNorm(vmin=1e-3, vmax=Hplot[~np.isnan(Hplot)].max()),
                   cmap='hot_r')
ax.contour(dt_ctrs, amp_ctrs, hot_cell.T.astype(float), levels=[0.5],
           colors='cyan', linewidths=1.2, linestyles='-')
ax.axvline(US_MASK_POST, color='white', lw=1.5, ls='--', label=f'L1窗口 {US_MASK_POST*1000:.0f}ms')
ax.set_xlabel('dt_from_US (s)')
ax.set_ylabel('振幅 (dB)')
ax.set_title('全通道 hits/cell/cycle (对数)')
ax.legend(fontsize=8, loc='upper right')
plt.colorbar(im, ax=ax, label='hits / cell / cycle')

# 密度热点掩膜
ax2 = axes1[1]
ax2.pcolormesh(dt_edges, amp_edges, hot_cell.T.astype(float), shading='flat',
               cmap='RdYlBu_r', vmin=0, vmax=1)
ax2.axvline(US_MASK_POST, color='black', lw=1.5, ls='--', label=f'L1边界 {US_MASK_POST*1000:.0f}ms')
ax2.set_xlabel('dt_from_US (s)')
ax2.set_ylabel('振幅 (dB)')
ax2.set_title(f'热点掩膜 (阈值={density_thresh:.4f})\n蓝=真实AE  红=干扰热点')
ax2.legend(fontsize=8)

# 振幅边缘分布对比（干扰 vs 真实AE）
ax3 = axes1[2]
amp_e2 = np.arange(35, 105, 2)
h_c, _  = np.histogram(ae_amp[contam],  bins=amp_e2)
h_cl, _ = np.histogram(ae_amp[~contam], bins=amp_e2)
amp_c2  = (amp_e2[:-1] + amp_e2[1:]) / 2
ax3.bar(amp_c2, h_c,  width=1.8, color='tomato',    alpha=0.7, label=f'干扰 ({n_contam})')
ax3.bar(amp_c2, h_cl, width=1.8, color='steelblue', alpha=0.7, label=f'真实AE ({n_clean})')
ax3.axvline(amp_ref, color='black', lw=1, ls=':', label=f'参考峰值 {amp_ref:.0f} dB')
ax3.set_xlabel('振幅 (dB)')
ax3.set_ylabel('频数')
ax3.set_title('振幅分布：干扰 vs 真实AE')
ax3.legend(fontsize=8)

plt.tight_layout()
out1 = os.path.join(RESULT_DIR, 'v4_01_2D密度干扰热点图.png')
fig1.savefig(out1)
plt.close(fig1)
print(f"图1已保存: {out1}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图2  Vp 演化（对零AIC校正）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), sharex='col')
fig2.suptitle(f'P波速度演化 (对零AIC校正)  t_cal={sys_delay:.3f} us\n'
              f'Vp = {H_MM:.0f}mm / (t_AIC - {sys_delay:.3f}us)',
              fontsize=12, fontweight='bold')
(ax2a, ax2b), (ax2c, ax2d) = axes2

ax2a.plot(us_ts, us_pt_sw,  color='navy',    lw=0.8, alpha=0.8, label='软件')
ax2a.plot(us_ts, us_pt_aic, color='crimson', lw=0.8, alpha=0.8, label='AIC')
ax2a.set_ylabel('P波到时 (us)')
ax2a.set_title('P波到时 (全段)')
ax2a.legend(); ax2a.yaxis.set_minor_locator(AutoMinorLocator())

ax2b.plot(us_ts, us_pt_aic - us_pt_sw, color='purple', lw=0.6, alpha=0.7)
ax2b.axhline(0, color='black', lw=0.8, ls='--')
ax2b.axhline(np.nanmean(diff_valid), color='red', lw=1, ls=':',
             label=f'均值 {np.nanmean(diff_valid):.2f} us')
ax2b.set_ylabel('AIC - 软件 (us)'); ax2b.set_title('拾取差值'); ax2b.legend()

vb = ~(np.isnan(us_Vp_aic) | np.isnan(us_Vp_sw))
ax2c.plot(us_ts[vb], us_Vp_sw[vb],  color='navy',    lw=0.8, alpha=0.8, label='Vp(软件)')
ax2c.plot(us_ts[vb], us_Vp_aic[vb], color='crimson', lw=0.8, alpha=0.8, label='Vp(AIC)')
ax2c.set_ylabel('Vp (km/s)'); ax2c.set_xlabel('时间 (s)')
ax2c.set_title('P波速度全段对比')
ax2c.legend(); ax2c.yaxis.set_minor_locator(AutoMinorLocator())

ax2d.scatter(us_Vp_sw[vb], us_Vp_aic[vb], s=3, alpha=0.4, color='steelblue')
lim = [min(np.nanmin(us_Vp_sw[vb]), np.nanmin(us_Vp_aic[vb]))*0.95,
       max(np.nanmax(us_Vp_sw[vb]), np.nanmax(us_Vp_aic[vb]))*1.05]
ax2d.plot(lim, lim, 'r--', lw=1, label='1:1')
ax2d.set_xlabel('Vp软件 (km/s)'); ax2d.set_ylabel('Vp AIC (km/s)')
ax2d.set_title('散点对比'); ax2d.legend()
ax2d.set_xlim(lim); ax2d.set_ylim(lim)

vline_fail([ax2a, ax2c], t_fail)
plt.tight_layout()
out2 = os.path.join(RESULT_DIR, 'v4_02_P波速度AIC.png')
fig2.savefig(out2)
plt.close(fig2)
print(f"图2已保存: {out2}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图3  全段干扰对比图（6通道）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("生成全段干扰对比图（6通道）...")
T_MAX  = ae_t.max() + 20
SZ, AL = 1.5, 0.40

fig3 = plt.figure(figsize=(20, 28))
gs3  = gridspec.GridSpec(6, 2, figure=fig3,
                         hspace=0.10, wspace=0.06,
                         top=0.93, bottom=0.04, left=0.07, right=0.97)
fig3.suptitle(
    f'声发射振幅 全段干扰对比 (v4)  0-{T_MAX:.0f} s\n'
    f'Layer-1时间窗({US_MASK_PRE*1000:.0f}ms/{US_MASK_POST*1000:.0f}ms) + '
    f'Layer-2密度热点(阈值={density_thresh:.4f} hits/cell/cycle)\n'
    f'去除 {n_contam} ({pct:.1f}%)  保留 {n_clean} 真实AE ({100-pct:.1f}%)',
    fontsize=11, fontweight='bold')

amp_ylim = [ae_amp.min()-5, ae_amp.max()+5]
from matplotlib.lines import Line2D

for ch in range(1, 7):
    row  = ch - 1
    d_ct = ae_contam[ae_contam['CH'] == ch]
    d_cl = ae_clean[ae_clean['CH']   == ch]

    axL = fig3.add_subplot(gs3[row, 0])
    if len(d_ct):
        axL.scatter(d_ct['Time'], d_ct['AMP'], s=SZ, alpha=AL*0.5,
                    color='silver', zorder=1, rasterized=True)
    if len(d_cl):
        axL.scatter(d_cl['Time'], d_cl['AMP'], s=SZ, alpha=AL,
                    color=CH_COLORS[ch], zorder=2, rasterized=True)
    axL.axhline(amp_ref, color='red', lw=0.8, ls=':', alpha=0.5,
                label=f'参考{amp_ref:.0f}dB')
    axL.set_ylabel(f'CH{ch}\n振幅(dB)', fontsize=9)
    axL.set_ylim(amp_ylim); axL.set_xlim(0, T_MAX)
    axL.yaxis.set_major_locator(MultipleLocator(20))
    axL.yaxis.set_minor_locator(MultipleLocator(10))
    if row == 0:
        axL.set_title(f'原始 ({n_hits} hits)', fontsize=11, pad=8)
        axL.legend(handles=[
            Line2D([0],[0], marker='o', ls='None', color='silver',
                   markersize=4, label=f'干扰 ({n_contam}, {pct:.1f}%)'),
            Line2D([0],[0], marker='o', ls='None', color=CH_COLORS[ch],
                   markersize=4, label=f'真实AE ({n_clean}, {100-pct:.1f}%)')],
            fontsize=8, loc='upper left')

    axR = fig3.add_subplot(gs3[row, 1], sharey=axL)
    if len(d_cl):
        axR.scatter(d_cl['Time'], d_cl['AMP'], s=SZ, alpha=AL,
                    color=CH_COLORS[ch], rasterized=True)
    axR.axhline(amp_ref, color='red', lw=0.6, ls=':', alpha=0.35)
    axR.set_xlim(0, T_MAX); axR.set_ylim(amp_ylim)
    axR.yaxis.set_major_locator(MultipleLocator(20))
    axR.tick_params(labelleft=False)
    if row == 0:
        axR.set_title(f'去干扰后 ({n_clean}, {100-pct:.1f}%)', fontsize=11, pad=8)

    if row < 5:
        axL.tick_params(labelbottom=False)
        axR.tick_params(labelbottom=False)
    else:
        axL.set_xlabel('时间 (s)'); axR.set_xlabel('时间 (s)')

    if not np.isnan(t_fail):
        axL.axvline(t_fail, color='red', lw=0.8, ls=':', alpha=0.6)
        axR.axvline(t_fail, color='red', lw=0.8, ls=':', alpha=0.6)

out3 = os.path.join(RESULT_DIR, 'v4_03_全段干扰对比图.png')
fig3.savefig(out3)
plt.close(fig3)
print(f"图3已保存: {out3}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图4  综合分析
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig4, axes4 = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
fig4.suptitle('超声波与声发射综合分析 v4\n'
              'Ultrasonic Vp (AIC calibrated) + AE (2D density interference removal)',
              fontsize=12, fontweight='bold')
ax4a, ax4b, ax4c, ax4d = axes4

m_vp = ~np.isnan(us_Vp_aic)
ax4a.plot(us_ts[m_vp], us_Vp_aic[m_vp], color='crimson', lw=1, alpha=0.85,
          label=f'Vp AIC (t_cal={sys_delay:.2f}us)')
ax4a.fill_between(us_ts[m_vp], us_Vp_aic[m_vp], alpha=0.12, color='crimson')
ax4a.set_ylabel('Vp (km/s)')
ax4a.set_title(f'P波速度演化  初始{np.nanmean(us_Vp_aic[valid_aic][:10]):.2f} km/s  峰值{np.nanmax(us_Vp_aic):.2f} km/s')
ax4a.yaxis.set_minor_locator(AutoMinorLocator()); ax4a.legend(loc='upper left')

for ch in range(1, 7):
    d = ae_clean[ae_clean['CH'] == ch]
    ax4b.scatter(d['Time'], d['AMP'], s=1.5, alpha=0.35,
                 color=CH_COLORS[ch], label=f'CH{ch}', rasterized=True)
ax4b.axhline(amp_ref, color='red', lw=0.8, ls=':', alpha=0.5, label=f'参考 {amp_ref:.0f} dB')
ax4b.set_ylabel('AE振幅 (dB)')
ax4b.set_title('声发射振幅 (v4 2D密度去干扰后)')
ax4b.legend(handles=[plt.Line2D([0],[0],marker='o',ls='None',color=CH_COLORS[i+1],
            markersize=5, label=f'CH{i+1}') for i in range(6)],
            ncol=6, fontsize=8, loc='upper left', framealpha=0.6)

ax4c.scatter(ae_clean['Time'], ae_clean['ABS_E'], s=1.5, alpha=0.3,
             color='darkred', rasterized=True)
ax4c.set_yscale('log')
ax4c.set_ylabel('绝对能量 (aJ)')
ax4c.set_title('声发射绝对能量 (去干扰后)')

ae_cl_s = ae_clean.sort_values('Time')
ae_or_s = ae.sort_values('Time')
ax4d.plot(ae_or_s['Time'], np.arange(1, n_hits+1),
          color='tomato', lw=1.2, ls='--', alpha=0.7, label=f'原始 ({n_hits})')
ax4d.plot(ae_cl_s['Time'], np.arange(1, n_clean+1),
          color='steelblue', lw=1.5, alpha=0.9, label=f'v4去干扰 ({n_clean})')
ax4d.set_ylabel('累计撞击数'); ax4d.set_xlabel('时间 (s)')
ax4d.set_title('累计撞击数对比'); ax4d.legend()

vline_fail(axes4, t_fail)
if not np.isnan(t_fail):
    axes4[0].text(t_fail+15, ax4a.get_ylim()[0]*1.02,
                  f'破坏 {t_fail:.0f}s', color='red', fontsize=9)

plt.tight_layout()
out4 = os.path.join(RESULT_DIR, 'v4_04_综合分析.png')
fig4.savefig(out4)
plt.close(fig4)
print(f"图4已保存: {out4}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图5  分层去除效果：按 dt 段分析被去除的 hit 特征
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
fig5.suptitle('分层干扰去除效果诊断\nLayer-1 vs Layer-2 特征对比',
              fontsize=12, fontweight='bold')

# 只有 L1 去除的
mask_L1_only = contam_L1 & ~contam_L2
# 只有 L2 去除的
mask_L2_only = contam_L2 & ~contam_L1
# 两层都命中的（L1∩L2 — dt ∈[0,POST] 且为热点）
mask_both = contam_L1 & contam_L2

groups = {
    'L1专属 (时间窗)': (mask_L1_only, 'tomato'),
    'L2专属 (密度)':   (mask_L2_only, 'darkorange'),
    '真实AE':          (~contam,       'steelblue'),
}

# 振幅分布对比
ax = axes5[0, 0]
for lbl, (msk, clr) in groups.items():
    h, e = np.histogram(ae_amp[msk], bins=np.arange(35, 105, 2))
    c = (e[:-1]+e[1:])/2
    ax.plot(c, h, color=clr, lw=1.5, label=f'{lbl} (n={msk.sum()})')
ax.set_xlabel('振幅 (dB)'); ax.set_ylabel('频数')
ax.set_title('振幅分布对比')
ax.legend(fontsize=8)

# dt 分布对比
ax = axes5[0, 1]
for lbl, (msk, clr) in groups.items():
    dt_sub = dt_from_us[msk & ~np.isnan(dt_from_us)]
    dt_sub = dt_sub[dt_sub < US_EXT_MAX]
    if len(dt_sub):
        h, e = np.histogram(dt_sub, bins=np.arange(0, US_EXT_MAX+DT_BIN_S, DT_BIN_S))
        c = (e[:-1]+e[1:])/2
        ax.plot(c, h, color=clr, lw=1.2, label=lbl)
ax.axvline(US_MASK_POST, color='black', lw=1, ls='--', label=f'L1边界 {US_MASK_POST*1000:.0f}ms')
ax.set_xlabel('dt_from_US (s)'); ax.set_ylabel('频数')
ax.set_title('dt 分布 (各类别)')
ax.legend(fontsize=8)

# DURATION 分布对比
ax = axes5[0, 2]
dur_bins = np.logspace(0, 6, 50)
for lbl, (msk, clr) in groups.items():
    d_sub = ae_dur[msk]; d_sub = d_sub[d_sub > 0]
    if len(d_sub):
        h, e = np.histogram(d_sub, bins=dur_bins)
        c = (e[:-1]*e[1:])**0.5
        ax.semilogx(c, h, color=clr, lw=1.2, label=lbl)
ax.set_xlabel('DURATION (us)'); ax.set_ylabel('频数')
ax.set_title('持续时间分布')
ax.legend(fontsize=8)

# 各通道去除比例
ax = axes5[1, 0]
for ch in range(1, 7):
    ch_mask = ae_ch == ch
    n_ch  = ch_mask.sum()
    n_L1c = (contam_L1 & ch_mask).sum()
    n_L2c = (contam_L2 & ~contam_L1 & ch_mask).sum()
    n_clc = (~contam & ch_mask).sum()
    ax.bar(ch-0.25, 100.*n_L1c/n_ch, 0.25, color='tomato', label='L1' if ch==1 else '')
    ax.bar(ch,      100.*n_L2c/n_ch, 0.25, color='orange',  label='L2' if ch==1 else '')
    ax.bar(ch+0.25, 100.*n_clc/n_ch, 0.25, color='steelblue', label='真实AE' if ch==1 else '')
ax.set_xlabel('通道'); ax.set_ylabel('比例 (%)')
ax.set_title('各通道干扰比例')
ax.set_xticks(range(1, 7))
ax.legend(fontsize=8)

# 时间分段内的干扰/真实 AE 比例
ax = axes5[1, 1]
t_bins = np.arange(0, ae_t.max()+50, 50)
t_ctrs = (t_bins[:-1]+t_bins[1:])/2
n_all_t, _  = np.histogram(ae_t,         bins=t_bins)
n_cont_t, _ = np.histogram(ae_t[contam], bins=t_bins)
n_clea_t, _ = np.histogram(ae_t[~contam],bins=t_bins)
ax.stackplot(t_ctrs, n_cont_t, n_clea_t, labels=['干扰','真实AE'],
             colors=['silver','steelblue'], alpha=0.7)
ax.set_xlabel('时间 (s)'); ax.set_ylabel('撞击数 / 50s')
ax.set_title('时间分段内干扰 vs 真实AE')
ax.legend(fontsize=8)
if not np.isnan(t_fail):
    ax.axvline(t_fail, color='red', ls='--', lw=1.2)

# 2D密度图（dt 0-2.5s 放大视图，只看 L2 段）
ax = axes5[1, 2]
H_L2 = H_per_cyc.copy()
H_L2[:bg_start_bin, :] = np.nan   # 只显示 L2 段
H_L2[H_L2 == 0] = np.nan
if not np.all(np.isnan(H_L2)):
    im5 = ax.pcolormesh(dt_edges, amp_edges, H_L2.T, shading='flat',
                        norm=LogNorm(vmin=max(density_thresh*0.1, 1e-4),
                                     vmax=np.nanmax(H_L2)),
                        cmap='hot_r')
    cs = ax.contour(dt_ctrs, amp_ctrs, hot_cell.T.astype(float),
                    levels=[0.5], colors='cyan', linewidths=1.2)
    plt.colorbar(im5, ax=ax, label='hits/cell/cycle')
ax.axhline(amp_ref, color='white', lw=1, ls=':', alpha=0.8, label=f'{amp_ref:.0f}dB')
ax.axvline(US_MASK_POST, color='white', lw=1.2, ls='--', alpha=0.7)
ax.set_xlabel('dt_from_US (s)'); ax.set_ylabel('振幅 (dB)')
ax.set_title(f'L2 段密度放大 (dt>{US_MASK_POST*1000:.0f}ms)\n青线=热点边界')
ax.legend(fontsize=8)

plt.tight_layout()
out5 = os.path.join(RESULT_DIR, 'v4_05_分层去除诊断.png')
fig5.savefig(out5)
plt.close(fig5)
print(f"图5已保存: {out5}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图6  AE 事件空间分布
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if len(evts) > 0 and 'x' in evts.columns:
    fig6, axes6 = plt.subplots(1, 3, figsize=(18, 7))
    fig6.suptitle(f'声发射事件空间分布 ({len(evts)} 事件)', fontsize=12, fontweight='bold')
    sc_kw = dict(c=evts['time'], cmap='plasma', s=4, alpha=0.5)
    axes6[0].scatter(evts['x'], evts['y'], **sc_kw)
    axes6[0].set_xlabel('x (mm)'); axes6[0].set_ylabel('y (mm)'); axes6[0].set_title('XY')
    sc = axes6[1].scatter(evts['x'], evts['z'], **sc_kw)
    axes6[1].set_xlabel('x (mm)'); axes6[1].set_ylabel('z (mm)'); axes6[1].set_title('XZ')
    axes6[2].scatter(evts['y'], evts['z'], **sc_kw)
    axes6[2].set_xlabel('y (mm)'); axes6[2].set_ylabel('z (mm)'); axes6[2].set_title('YZ')
    plt.colorbar(sc, ax=axes6[2], label='时间 (s)', shrink=0.8)
    plt.tight_layout()
    out6 = os.path.join(RESULT_DIR, 'v4_06_AE事件空间分布.png')
    fig6.savefig(out6); plt.close(fig6)
    print(f"图6已保存: {out6}")
else:
    out6 = None

# ─── 保存 CSV ────────────────────────────────────────────────────────────
df_vp = pd.DataFrame({
    'time_s':        us_ts,
    'pwave_sw_us':   us_pt_sw,
    'pwave_aic_us':  us_pt_aic,
    'travel_aic_us': us_pt_aic - sys_delay,
    'Vp_sw_km':      us_Vp_sw,
    'Vp_aic_km':     us_Vp_aic,
})
out_vp = os.path.join(RESULT_DIR, 'v4_Vp_AIC.csv')
df_vp.to_csv(out_vp, index=False)

out_ae = os.path.join(RESULT_DIR, 'v4_AE_clean.csv')
ae_clean.to_csv(out_ae, index=False)

out_contam = os.path.join(RESULT_DIR, 'v4_AE_contaminated.csv')
ae_contam.to_csv(out_contam, index=False)

# ─── 统计汇总 ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("统计汇总 | Summary")
print("="*60)
print(f"\n[超声波 P波速度]")
print(f"  系统延时 t_cal:    {sys_delay:.3f} us")
print(f"  初始 Vp (前10):    {np.nanmean(us_Vp_aic[valid_aic][:10]):.3f} km/s")
print(f"  峰值 Vp:           {np.nanmax(us_Vp_aic):.3f} km/s")
if not np.isnan(t_fail):
    pfi = np.where(fail_mask)[0][0]
    print(f"  破坏前5次均值:     {np.nanmean(us_Vp_aic[max(0,pfi-5):pfi]):.3f} km/s")
    print(f"  推测破坏时刻:      {t_fail:.1f} s")
print(f"  AIC-软件 差值均值: {np.nanmean(diff_valid):.3f} us (sigma={np.nanstd(diff_valid):.3f})")

print(f"\n[声发射干扰识别 (v4 2D密度法)]")
print(f"  Layer-1 基础时间窗:  {n_L1} hits ({100.*n_L1/n_hits:.1f}%)")
print(f"  Layer-2 密度热点:    {n_L2} hits ({100.*n_L2/n_hits:.1f}%)")
print(f"  合计干扰:            {n_contam} hits ({pct:.1f}%)")
print(f"  真实AE:              {n_clean} hits ({100-pct:.1f}%)")
print(f"  vs v3 (58.2%去除):  本版去除{pct:.1f}%，"
      f"{'少去除' if pct < 58.2 else '多去除'}{abs(pct-58.2):.1f}%")

print(f"\n[输出目录] {RESULT_DIR}")
for f in [out1, out2, out3, out4, out5, out_vp, out_ae, out_contam] + ([out6] if out6 else []):
    print(f"  {os.path.basename(f)}")

print("\n分析完成！")
