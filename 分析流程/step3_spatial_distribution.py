#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3_spatial_distribution.py — AE 事件空间分布可视化
======================================================
读取仪器定位事件文件，与手动清理后（或 step1 自动滤波后）的干净 hits 交叉匹配，
仅展示干净事件的空间分布。输出：

  step3_2D空间分布.png   — 3 种 2D 投影（俯视 X-Z / 正视 X-Y / 侧视 Z-Y）
                           左列按时间着色，右列按震源振幅着色
  step3_3D空间分布.png   — 3 个视角的 3D 散点图，按时间着色
  step3_events_clean.csv — 过滤后干净事件表

运行方式:
  python step3_spatial_distribution.py
  （step1 完成后即可运行；step2 若已完成会自动使用更干净的结果）

坐标系（仪器定义，mm）:
  Y — 试样轴向（0 = 底端，H_MM = 顶端）
  X, Z — 试样横截面平面（原点 ≈ 截面中心）
"""

import sys, io, os, re
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    AE_EVTS,
    STEP2_AE_CLEAN, STEP1_AE_CLEAN,
    STEP3_DIR,
    H_MM, D_MM,
    EVT_Q_MIN, EVT_CLEAN_FRAC,
    B_WINDOW_N, B_STEP_N, B_AMP_MIN,
    CH_COLORS, SAVE_DPI,
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

for _fn in ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(_fn, fallback_to_default=False)
        plt.rcParams['font.family'] = _fn
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid']          = True
plt.rcParams['grid.alpha']         = 0.20
plt.rcParams['grid.linewidth']     = 0.4


# ═══════════════════════════════════════════════════════════════════════════
# § 1  解析仪器事件文件
# ═══════════════════════════════════════════════════════════════════════════
_GP_RE = re.compile(
    r'\* Gp#.*?x,y,z\s*=\s*([+-]?[\d.]+),\s*([+-]?[\d.]+),\s*([+-]?[\d.]+)'
    r'.*?q\s*=\s*([+-]?[\d.]+)'
    r'.*?Src Amplitude\s*=\s*([+-]?[\d.]+)'
)


def parse_ae_events(path: str) -> tuple[list[dict], list[dict]]:
    """
    解析仪器事件 TXT，返回 (events, all_hit_rows)。
    events: list of dict  {x,y,z,q,src_amp, hit_times:list, hit_chs:list}
    all_hit_rows: 文件中所有 hit 行（可用于重建未定位信号）
    """
    events: list[dict] = []
    cur: dict | None   = None

    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped.startswith('*'):
                continue
            m = _GP_RE.search(stripped)
            if m:
                if cur is not None:
                    events.append(cur)
                cur = dict(
                    x=float(m.group(1)), y=float(m.group(2)), z=float(m.group(3)),
                    q=float(m.group(4)), src_amp=float(m.group(5)),
                    hit_times=[], hit_chs=[], hit_abs_e=[]
                )
            elif cur is not None:
                # 解析 hit 行: * TIME CH RISE COUN ENER DUR AMP ABS_E
                parts = stripped.split()
                if len(parts) >= 9:
                    try:
                        cur['hit_times'].append(float(parts[1]))
                        cur['hit_chs'].append(int(parts[2]))
                        cur['hit_abs_e'].append(float(parts[8]))
                    except (ValueError, IndexError):
                        pass
    if cur is not None:
        events.append(cur)
    return events


# ═══════════════════════════════════════════════════════════════════════════
# § 2  与干净 hits 交叉匹配，生成事件 DataFrame
# ═══════════════════════════════════════════════════════════════════════════
def build_events_df(events: list[dict],
                    clean_df: pd.DataFrame,
                    q_min: float = EVT_Q_MIN,
                    clean_frac: float = EVT_CLEAN_FRAC) -> pd.DataFrame:
    """
    构建事件 DataFrame，并计算每个事件的干净 hits 占比。
    保留：q >= q_min  且  干净 hits 占比 >= clean_frac
    """
    # 干净 hits 查找表：(round(t,6), ch)
    clean_set: set[tuple] = set(
        zip(clean_df['Time'].round(6).astype(float),
            clean_df['CH'].astype(int))
    )

    rows = []
    for ev in events:
        ht = ev['hit_times']
        hc = ev['hit_chs']
        he = ev['hit_abs_e']
        n  = len(ht)
        if n == 0:
            continue
        t_first = ht[0]
        clean_mask = [(round(t, 6), ch) in clean_set for t, ch in zip(ht, hc)]
        n_clean    = sum(clean_mask)
        frac       = n_clean / n
        total_abs_e = sum(e for e, ok in zip(he, clean_mask) if ok)
        rows.append(dict(
            Time      = t_first,
            x         = ev['x'],
            y         = ev['y'],
            z         = ev['z'],
            q         = ev['q'],
            src_amp   = ev['src_amp'],
            total_abs_e = total_abs_e,
            n_hits    = n,
            n_clean   = n_clean,
            clean_f   = frac,
            is_clean  = (ev['q'] >= q_min) and (frac >= clean_frac),
        ))

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values('Time').reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# § 3  绘图辅助函数
# ═══════════════════════════════════════════════════════════════════════════
def _outline_top(ax, r):
    """俯视图（X-Z 平面）圆形轮廓。"""
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(r*np.cos(th), r*np.sin(th),
            color='#666666', ls='--', lw=0.9, alpha=0.6, zorder=0)


def _outline_side(ax, r, h):
    """正/侧视图（* -Y 平面）矩形轮廓。"""
    from matplotlib.patches import Rectangle
    rect = Rectangle((-r, 0), 2*r, h,
                     fill=False, edgecolor='#666666',
                     linestyle='--', linewidth=0.9, alpha=0.6, zorder=0)
    ax.add_patch(rect)


def _cylinder_3d(ax, r, h, n_circ=120, n_vert=8,
                 color='#888888', alpha=0.25, lw=0.8):
    """在 3D 轴上绘制试样圆柱骨架（两端圆 + 竖向母线）。"""
    th = np.linspace(0, 2*np.pi, n_circ)
    ax.plot(r*np.cos(th), r*np.sin(th), np.zeros(n_circ),
            color=color, lw=lw, alpha=alpha)
    ax.plot(r*np.cos(th), r*np.sin(th), np.full(n_circ, h),
            color=color, lw=lw, alpha=alpha)
    for t in np.linspace(0, 2*np.pi, n_vert, endpoint=False):
        ax.plot([r*np.cos(t), r*np.cos(t)],
                [r*np.sin(t), r*np.sin(t)],
                [0, h],
                color=color, lw=lw, alpha=alpha)


def _scatter_cbar(fig, ax, x, y, c, cmap, vmin, vmax,
                  cbar_label, xlabel, ylabel, title, r, h, is_top=False):
    """统一绘制 2D 投影子图 + 色条。"""
    sc = ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=10, alpha=0.65, edgecolors='none', zorder=2, rasterized=True)
    if is_top:
        _outline_top(ax, r)
        ax.set_aspect('equal', 'box')
    else:
        _outline_side(ax, r, h)
        ax.set_ylim(-5, h+5)
        ax.set_xlim(-r*1.5, r*1.5)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, pad=4)
    ax.grid(True, alpha=0.2, linewidth=0.4)


# ═══════════════════════════════════════════════════════════════════════════
# § 4  生成 2D 分布图
# ═══════════════════════════════════════════════════════════════════════════
def gen_2d_figure(df: pd.DataFrame, out_dir: str,
                  h: float = H_MM, d: float = D_MM) -> str:
    r = d / 2.0
    n = len(df)

    t_norm  = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min() + 1e-9)
    t_vals  = df['Time'].values
    a_vals  = df['src_amp'].values
    x, y, z = df['x'].values, df['y'].values, df['z'].values

    t_min, t_max = df['Time'].min(), df['Time'].max()
    a_min, a_max = a_vals.min(), a_vals.max()

    plt.ioff()
    fig = plt.figure(figsize=(16, 20))
    try:
        fig.suptitle(
            f'AE 事件空间分布（2D 投影）  —  {n} 个定位事件\n'
            f'时间范围: {t_min:.1f} ~ {t_max:.1f} s  '
            f'震源振幅: {a_min:.0f} ~ {a_max:.0f} dB',
            fontsize=12, fontweight='bold'
        )
        gs = gridspec.GridSpec(3, 2, figure=fig,
                               hspace=0.38, wspace=0.34,
                               top=0.91, bottom=0.05,
                               left=0.09, right=0.97)

        # ── 行 0：俯视图 X-Z（Y 轴向上 = 进纸面）────────────────────
        ax00 = fig.add_subplot(gs[0, 0])
        _scatter_cbar(fig, ax00, x, z, t_vals, 'viridis', t_min, t_max,
                      '时间 (s)', 'X (mm)', 'Z (mm)',
                      '俯视（X-Z，顶端向上）\n按时间着色', r, h, is_top=True)

        ax01 = fig.add_subplot(gs[0, 1])
        _scatter_cbar(fig, ax01, x, z, a_vals, 'hot_r', a_min, a_max,
                      '震源振幅 (dB)', 'X (mm)', 'Z (mm)',
                      '俯视（X-Z，顶端向上）\n按震源振幅着色', r, h, is_top=True)

        # ── 行 1：正视图 X-Y（X 水平，Y 高度）───────────────────────
        ax10 = fig.add_subplot(gs[1, 0])
        _scatter_cbar(fig, ax10, x, y, t_vals, 'viridis', t_min, t_max,
                      '时间 (s)', 'X (mm)', 'Y (mm，高度)',
                      '正视（X-Y）\n按时间着色', r, h)
        ax10.yaxis.set_major_locator(MultipleLocator(20))

        ax11 = fig.add_subplot(gs[1, 1])
        _scatter_cbar(fig, ax11, x, y, a_vals, 'hot_r', a_min, a_max,
                      '震源振幅 (dB)', 'X (mm)', 'Y (mm，高度)',
                      '正视（X-Y）\n按震源振幅着色', r, h)
        ax11.yaxis.set_major_locator(MultipleLocator(20))

        # ── 行 2：侧视图 Z-Y（Z 水平，Y 高度）───────────────────────
        ax20 = fig.add_subplot(gs[2, 0])
        _scatter_cbar(fig, ax20, z, y, t_vals, 'viridis', t_min, t_max,
                      '时间 (s)', 'Z (mm)', 'Y (mm，高度)',
                      '侧视（Z-Y）\n按时间着色', r, h)
        ax20.yaxis.set_major_locator(MultipleLocator(20))

        ax21 = fig.add_subplot(gs[2, 1])
        _scatter_cbar(fig, ax21, z, y, a_vals, 'hot_r', a_min, a_max,
                      '震源振幅 (dB)', 'Z (mm)', 'Y (mm，高度)',
                      '侧视（Z-Y）\n按震源振幅着色', r, h)
        ax21.yaxis.set_major_locator(MultipleLocator(20))

        out = os.path.join(out_dir, 'step3_2D空间分布.png')
        fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
        return out
    finally:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# § 5  生成 3D 分布图
# ═══════════════════════════════════════════════════════════════════════════
def gen_3d_figure(df: pd.DataFrame, out_dir: str,
                  h: float = H_MM, d: float = D_MM) -> str:
    r = d / 2.0
    n = len(df)

    t_vals  = df['Time'].values
    a_vals  = df['src_amp'].values
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    t_min, t_max = t_vals.min(), t_vals.max()
    a_min, a_max = a_vals.min(), a_vals.max()

    # 时间归一化为颜色
    t_norm = (t_vals - t_min) / (t_max - t_min + 1e-9)

    views = [
        (25,  45,  '视角①（方位 45°）'),
        (25, -45,  '视角②（方位 -45°）'),
        (60,  30,  '视角③（俯瞰，仰角 60°）'),
    ]

    plt.ioff()
    # ── Figure A: 3×1 按时间着色 ────────────────────────────────────────
    figA = plt.figure(figsize=(21, 8))
    try:
        figA.suptitle(
            f'AE 事件空间分布（3D，按时间着色）  —  {n} 个定位事件\n'
            f'早期（深色）→ 晚期（浅色）　颜色图例：viridis',
            fontsize=12, fontweight='bold'
        )
        cmap_t = plt.cm.viridis
        sm_t   = plt.cm.ScalarMappable(cmap=cmap_t,
                                        norm=plt.Normalize(t_min, t_max))
        sm_t.set_array([])

        for i, (elev, azim, title) in enumerate(views):
            ax = figA.add_subplot(1, 3, i+1, projection='3d')
            ax.scatter(x, z, y,
                       c=t_norm, cmap=cmap_t,
                       s=12, alpha=0.65, edgecolors='none',
                       depthshade=True, rasterized=True)
            _cylinder_3d(ax, r, h)
            ax.set_xlabel('X (mm)', fontsize=8, labelpad=2)
            ax.set_ylabel('Z (mm)', fontsize=8, labelpad=2)
            ax.set_zlabel('Y 高度 (mm)', fontsize=8, labelpad=2)
            ax.set_zlim(0, h)
            ax.set_title(title, fontsize=9, pad=4)
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(labelsize=7)
            ax.zaxis.set_major_locator(MultipleLocator(20))

        cb = figA.colorbar(sm_t, ax=figA.axes, shrink=0.55, pad=0.04,
                           orientation='vertical', fraction=0.018)
        cb.set_label('时间 (s)', fontsize=9)
        cb.ax.tick_params(labelsize=8)

        outA = os.path.join(out_dir, 'step3_3D空间分布_时间.png')
        figA.savefig(outA, dpi=SAVE_DPI, bbox_inches='tight')
    finally:
        plt.close(figA)

    # ── Figure B: 3×1 按震源振幅着色 ────────────────────────────────────
    plt.ioff()
    figB = plt.figure(figsize=(21, 8))
    try:
        figB.suptitle(
            f'AE 事件空间分布（3D，按震源振幅着色）  —  {n} 个定位事件\n'
            f'低振幅（深色）→ 高振幅（浅色）　颜色图例：hot_r',
            fontsize=12, fontweight='bold'
        )
        cmap_a = plt.cm.hot_r
        sm_a   = plt.cm.ScalarMappable(cmap=cmap_a,
                                        norm=plt.Normalize(a_min, a_max))
        sm_a.set_array([])
        a_norm = (a_vals - a_min) / (a_max - a_min + 1e-9)

        for i, (elev, azim, title) in enumerate(views):
            ax = figB.add_subplot(1, 3, i+1, projection='3d')
            ax.scatter(x, z, y,
                       c=a_norm, cmap=cmap_a,
                       s=12, alpha=0.65, edgecolors='none',
                       depthshade=True, rasterized=True)
            _cylinder_3d(ax, r, h)
            ax.set_xlabel('X (mm)', fontsize=8, labelpad=2)
            ax.set_ylabel('Z (mm)', fontsize=8, labelpad=2)
            ax.set_zlabel('Y 高度 (mm)', fontsize=8, labelpad=2)
            ax.set_zlim(0, h)
            ax.set_title(title, fontsize=9, pad=4)
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(labelsize=7)
            ax.zaxis.set_major_locator(MultipleLocator(20))

        cb = figB.colorbar(sm_a, ax=figB.axes, shrink=0.55, pad=0.04,
                           orientation='vertical', fraction=0.018)
        cb.set_label('震源振幅 (dB)', fontsize=9)
        cb.ax.tick_params(labelsize=8)

        outB = os.path.join(out_dir, 'step3_3D空间分布_振幅.png')
        figB.savefig(outB, dpi=SAVE_DPI, bbox_inches='tight')
    finally:
        plt.close(figB)

    return outA, outB


# ═══════════════════════════════════════════════════════════════════════════
# § 6  生成时间演化图（分段着色：早 / 中 / 晚三期）
# ═══════════════════════════════════════════════════════════════════════════
# § 5b 生成 3D 能量分布图
# ═══════════════════════════════════════════════════════════════════════════
def gen_3d_energy_figure(df: pd.DataFrame, out_dir: str,
                         h: float = H_MM, d: float = D_MM) -> str:
    """3 视角 3D 散点，按事件累计绝对能量（aJ）着色，点大小也随能量变化。"""
    r = d / 2.0
    n = len(df)

    x, y, z  = df['x'].values, df['y'].values, df['z'].values
    e_vals   = df['total_abs_e'].values
    e_min, e_max = e_vals.min(), e_vals.max()

    # 点大小：能量线性映射到 6~80 px²
    e_norm = (e_vals - e_min) / (e_max - e_min + 1e-9)
    sizes  = 6 + e_norm * 74

    views = [
        (25,  45,  '视角①（方位 45°）'),
        (25, -45,  '视角②（方位 -45°）'),
        (60,  30,  '视角③（俯瞰，仰角 60°）'),
    ]

    plt.ioff()
    fig = plt.figure(figsize=(21, 8))
    try:
        fig.suptitle(
            f'AE 事件空间分布（3D，按累计绝对能量着色）  —  {n} 个定位事件\n'
            f'低能量（深色）→ 高能量（浅色）　颜色图例：YlOrRd　点大小正比于能量',
            fontsize=12, fontweight='bold'
        )
        cmap_e = plt.cm.YlOrRd
        sm_e   = plt.cm.ScalarMappable(cmap=cmap_e,
                                        norm=plt.Normalize(e_min, e_max))
        sm_e.set_array([])

        for i, (elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            sc = ax.scatter(x, z, y,
                            c=e_norm, cmap=cmap_e,
                            s=sizes, alpha=0.75,
                            edgecolors='#333333', linewidths=0.2,
                            depthshade=True, rasterized=True)
            _cylinder_3d(ax, r, h)
            ax.set_xlabel('X (mm)', fontsize=8, labelpad=2)
            ax.set_ylabel('Z (mm)', fontsize=8, labelpad=2)
            ax.set_zlabel('Y 高度 (mm)', fontsize=8, labelpad=2)
            ax.set_zlim(0, h)
            ax.set_title(title, fontsize=9, pad=4)
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(labelsize=7)
            ax.zaxis.set_major_locator(MultipleLocator(20))

        cb = fig.colorbar(sm_e, ax=fig.axes, shrink=0.55, pad=0.04,
                          orientation='vertical', fraction=0.018)
        cb.set_label('累计绝对能量 (aJ)', fontsize=9)
        cb.ax.tick_params(labelsize=8)

        out = os.path.join(out_dir, 'step3_3D空间分布_能量.png')
        fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
        return out
    finally:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
def gen_evolution_figure(df: pd.DataFrame, out_dir: str,
                         h: float = H_MM, d: float = D_MM) -> str:
    """
    将事件按时间三等分（早期 / 中期 / 临近破坏），
    在同一 3D 图上用三种颜色分别绘制，直观展示裂缝演化。
    """
    r = d / 2.0
    n = len(df)
    t_vals = df['Time'].values
    t33, t67 = np.percentile(t_vals, [33, 67])

    mask_early  = t_vals <= t33
    mask_mid    = (t_vals > t33) & (t_vals <= t67)
    mask_late   = t_vals > t67

    stages = [
        (mask_early, '#2196F3', f'早期  ({mask_early.sum()} 个事件,  t≤{t33:.0f}s)'),
        (mask_mid,   '#FF9800', f'中期  ({mask_mid.sum()} 个事件,  {t33:.0f}s<t≤{t67:.0f}s)'),
        (mask_late,  '#F44336', f'临近破坏  ({mask_late.sum()} 个事件,  t>{t67:.0f}s)'),
    ]
    x, y, z = df['x'].values, df['y'].values, df['z'].values

    views = [(25, 45), (25, -45), (60, 30)]
    view_titles = ['视角① (方位 45°)', '视角② (方位 -45°)', '视角③ (俯瞰)']

    plt.ioff()
    fig = plt.figure(figsize=(21, 8))
    try:
        fig.suptitle(
            f'AE 事件时间演化（三阶段空间分布）  —  {n} 个定位事件\n'
            '蓝=早期 / 橙=中期 / 红=临近破坏',
            fontsize=12, fontweight='bold'
        )
        for i, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            for mask, color, label in stages:
                if mask.sum() == 0:
                    continue
                ax.scatter(x[mask], z[mask], y[mask],
                           c=color, s=12, alpha=0.65,
                           edgecolors='none', depthshade=True,
                           rasterized=True, label=label)
            _cylinder_3d(ax, r, h)
            ax.set_xlabel('X (mm)', fontsize=8, labelpad=2)
            ax.set_ylabel('Z (mm)', fontsize=8, labelpad=2)
            ax.set_zlabel('Y 高度 (mm)', fontsize=8, labelpad=2)
            ax.set_zlim(0, h)
            ax.set_title(view_titles[i], fontsize=9, pad=4)
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(labelsize=7)
            ax.zaxis.set_major_locator(MultipleLocator(20))
            if i == 0:
                ax.legend(loc='upper left', fontsize=7,
                          bbox_to_anchor=(-0.05, 1.05),
                          framealpha=0.8)

        out = os.path.join(out_dir, 'step3_3D时间演化.png')
        fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
        return out
    finally:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# § 7  动态 b 值分析
# ═══════════════════════════════════════════════════════════════════════════
def _moving_avg(arr: np.ndarray, window: int) -> np.ndarray:
    """均匀滑动平均，边界用 same 模式填充。"""
    if window <= 1 or len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def _mle_b(amps: np.ndarray, amp_min: float) -> tuple:
    """
    最大似然估计 b 值（Aki 1965）及其标准误差。
    适用于 AE 振幅（dB），内部将 AMP/20 作为震级。

    b = 20·log₁₀(e) / (mean_AMP - amp_min)
    sigma_b = b / sqrt(N)
    """
    a = amps[amps >= amp_min]
    n = len(a)
    if n < 5:
        return np.nan, np.nan
    delta = a.mean() - amp_min
    if delta <= 0:
        return np.nan, np.nan
    b     = 20.0 * np.log10(np.e) / delta
    sigma = b / np.sqrt(n)
    return b, sigma


def compute_b_series(hits_df: pd.DataFrame) -> pd.DataFrame:
    """
    对干净 hits 按时间顺序做滑动窗口 MLE b 值计算。
    返回 DataFrame: Time（窗口中点时刻）, b, sigma_b
    """
    amps  = hits_df['AMP'].values.astype(float)
    times = hits_df['Time'].values.astype(float)
    n_tot = len(amps)
    rows  = []
    for start in range(0, n_tot - B_WINDOW_N + 1, B_STEP_N):
        end   = start + B_WINDOW_N
        t_mid = times[start + B_WINDOW_N // 2]
        b, sigma = _mle_b(amps[start:end], B_AMP_MIN)
        rows.append(dict(Time=t_mid, b=b, sigma_b=sigma))
    return pd.DataFrame(rows)


def gen_b_value_figure(b_df: pd.DataFrame,
                       hits_df: pd.DataFrame,
                       out_dir: str) -> str:
    """
    综合 b 值分析图，4 行布局：
      行0: AE 振幅散点（背景参考，按通道着色）
      行1: b 值时间序列（主图，含 ±1σ/±2σ 误差带和平滑曲线）
      行2: AE hits 速率（30s 分箱）
      行3: 三阶段 G-R 分布（早期 / 中期 / 临近破坏）
    """
    b_valid = b_df.dropna(subset=['b']).copy()
    n_hits  = len(hits_df)
    n_b     = len(b_valid)

    t_arr   = hits_df['Time'].values.astype(float)
    t_min_h = t_arr.min()
    t_max_h = t_arr.max()
    x_range = (max(0, t_min_h - 10), t_max_h + 10)

    # 平滑 b 值（取窗口数的 1/15，最少 3）
    smooth_w = max(3, n_b // 15)
    b_vals   = b_valid['b'].values
    b_smooth = _moving_avg(b_vals, smooth_w)
    t_b      = b_valid['Time'].values
    s_b      = b_valid['sigma_b'].values

    # AE 速率分箱
    bin_w  = 30.0
    t_bins = np.arange(t_min_h, t_max_h + bin_w, bin_w)
    rate_counts, _ = np.histogram(t_arr, bins=t_bins)
    rate_t  = 0.5 * (t_bins[:-1] + t_bins[1:])
    rate_sm = _moving_avg(rate_counts.astype(float), 5)

    # 三阶段边界（按 hits 时间三等分）
    t33, t67 = np.percentile(t_arr, [33, 67])
    stage_colors  = ['#2196F3', '#FF9800', '#F44336']
    stage_labels  = ['早期', '中期', '临近破坏']
    stage_masks   = [
        hits_df['AMP'].index[t_arr <= t33],
        hits_df['AMP'].index[(t_arr > t33) & (t_arr <= t67)],
        hits_df['AMP'].index[t_arr > t67],
    ]
    # 直接用布尔
    m_early = t_arr <= t33
    m_mid   = (t_arr > t33) & (t_arr <= t67)
    m_late  = t_arr > t67

    plt.ioff()
    fig = plt.figure(figsize=(16, 22))
    try:
        fig.suptitle(
            f'动态 b 值分析（MLE，滑动窗口）\n'
            f'数据来源: {n_hits} hits  |  窗口 {B_WINDOW_N} hits  步长 {B_STEP_N} hits  '
            f'完整性阈值 {B_AMP_MIN:.0f} dB  |  共 {n_b} 个计算点',
            fontsize=11, fontweight='bold'
        )
        gs = gridspec.GridSpec(
            4, 3, figure=fig,
            height_ratios=[1.0, 2.8, 1.0, 1.8],
            hspace=0.40, wspace=0.30,
            top=0.92, bottom=0.04, left=0.09, right=0.97
        )

        # ── 行0: 振幅散点（背景参考）────────────────────────────────
        ax0 = fig.add_subplot(gs[0, :])
        for ch in range(1, 7):
            sub = hits_df[hits_df['CH'] == ch]
            if len(sub):
                ax0.scatter(sub['Time'], sub['AMP'],
                            s=1.0, alpha=0.25, color=CH_COLORS[ch],
                            rasterized=True, label=f'CH{ch}')
        ax0.axvline(t33, color='#888888', lw=0.8, ls='--')
        ax0.axvline(t67, color='#888888', lw=0.8, ls='--')
        ax0.set_xlim(*x_range)
        ax0.set_ylabel('振幅 (dB)', fontsize=9)
        ax0.set_title('AE hits 振幅散点（背景参考）', fontsize=10, pad=3)
        ax0.yaxis.set_major_locator(MultipleLocator(20))
        ax0.tick_params(labelbottom=False)
        ax0.legend(ncol=6, fontsize=7, loc='upper left',
                   markerscale=5, framealpha=0.7)
        # 阶段背景
        for col, (lo, hi) in zip(stage_colors,
                                  [(x_range[0], t33), (t33, t67), (t67, x_range[1])]):
            ax0.axvspan(lo, hi, alpha=0.05, color=col)

        # ── 行1: b 值时间序列（主图）────────────────────────────────
        ax1 = fig.add_subplot(gs[1, :])
        if n_b >= 2:
            # ±2σ 浅带
            ax1.fill_between(t_b, b_vals - 2*s_b, b_vals + 2*s_b,
                             alpha=0.10, color='#1565C0', label='±2σ')
            # ±1σ 带
            ax1.fill_between(t_b, b_vals - s_b, b_vals + s_b,
                             alpha=0.22, color='#1565C0', label='±1σ')
            # 原始细线
            ax1.plot(t_b, b_vals, color='#90CAF9', lw=0.5, alpha=0.55)
            # 平滑粗线
            b_mean = float(np.nanmean(b_vals))
            ax1.plot(t_b, b_smooth, color='#0D47A1', lw=2.2,
                     label=f'b 值（平滑，均值 {b_mean:.3f}）')
            # 参考线
            for ref, lbl in [(1.0, 'b=1.0'), (1.5, 'b=1.5'), (0.5, 'b=0.5')]:
                ax1.axhline(ref, color='#999999', lw=0.7, ls='--', alpha=0.6)
                ax1.text(x_range[1] - 8, ref + 0.02, lbl,
                         fontsize=7, color='#777777', ha='right')
            # 均值线
            ax1.axhline(b_mean, color='#E53935', lw=1.2, ls='-.',
                        alpha=0.85, label=f'总体均值 {b_mean:.3f}')
            # 自动 y 轴范围
            b_lo = max(0, float(np.nanpercentile(b_vals, 1)) - 0.4)
            b_hi = float(np.nanpercentile(b_vals, 99)) + 0.4
            ax1.set_ylim(b_lo, b_hi)

        ax1.axvline(t33, color='#888888', lw=0.8, ls='--')
        ax1.axvline(t67, color='#888888', lw=0.8, ls='--')
        for col, lbl, tx in zip(stage_colors, stage_labels,
                                 [t33*0.5, (t33+t67)*0.5, (t67+t_max_h)*0.5]):
            ax1.text(tx, ax1.get_ylim()[1] * 0.97 if n_b >= 2 else 1.5,
                     lbl, color=col, fontsize=9, ha='center',
                     fontweight='bold', va='top')
        for col, (lo, hi) in zip(stage_colors,
                                  [(x_range[0], t33), (t33, t67), (t67, x_range[1])]):
            ax1.axvspan(lo, hi, alpha=0.04, color=col)

        ax1.set_xlim(*x_range)
        ax1.set_ylabel('b 值', fontsize=11)
        ax1.set_title('b 值动态演化（MLE，滑动窗口）', fontsize=11, pad=5)
        ax1.legend(ncol=4, fontsize=8, loc='upper right', framealpha=0.85)
        ax1.tick_params(labelbottom=False)
        ax1.grid(True, alpha=0.18, linewidth=0.4)

        # ── 行2: AE hits 速率────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2, :])
        ax2.bar(rate_t, rate_counts, width=bin_w*0.85,
                color='#BBDEFB', edgecolor='none', alpha=0.65)
        ax2.plot(rate_t, rate_sm, color='#1976D2', lw=1.5)
        ax2.axvline(t33, color='#888888', lw=0.8, ls='--')
        ax2.axvline(t67, color='#888888', lw=0.8, ls='--')
        for col, (lo, hi) in zip(stage_colors,
                                  [(x_range[0], t33), (t33, t67), (t67, x_range[1])]):
            ax2.axvspan(lo, hi, alpha=0.05, color=col)
        ax2.set_xlim(*x_range)
        ax2.set_xlabel('时间 (s)', fontsize=9)
        ax2.set_ylabel(f'计数/{bin_w:.0f}s', fontsize=9)
        ax2.set_title(f'AE hits 速率（{bin_w:.0f}s 分箱）', fontsize=10, pad=3)
        ax2.grid(True, alpha=0.18, linewidth=0.4)

        # ── 行3: 三阶段 G-R 分布──────────────────────────────────────
        stage_data = [
            (m_early, stage_colors[0], stage_labels[0]),
            (m_mid,   stage_colors[1], stage_labels[1]),
            (m_late,  stage_colors[2], stage_labels[2]),
        ]
        for s_i, (mask, col, lbl) in enumerate(stage_data):
            ax3 = fig.add_subplot(gs[3, s_i])
            amps_s = hits_df['AMP'].values[mask]
            amps_s = amps_s[amps_s >= B_AMP_MIN]
            if len(amps_s) < 10:
                ax3.text(0.5, 0.5, '数据不足', ha='center', va='center',
                         transform=ax3.transAxes, fontsize=9)
                ax3.set_title(f'G-R 分布: {lbl}', fontsize=9, color=col)
                continue

            # 累积频次
            amp_uniq = np.sort(np.unique(amps_s))
            n_cum    = np.array([np.sum(amps_s >= a) for a in amp_uniq])
            log_n    = np.log10(n_cum.astype(float))

            ax3.scatter(amp_uniq, log_n, s=22, color=col, alpha=0.85,
                        zorder=3)

            # 最小二乘拟合
            if len(amp_uniq) >= 3:
                coeffs = np.polyfit(amp_uniq, log_n, 1)
                x_fit  = np.linspace(amp_uniq[0], amp_uniq[-1], 200)
                y_fit  = np.polyval(coeffs, x_fit)
                b_ls   = -coeffs[0] * 20   # b = -斜率 × 20
                ax3.plot(x_fit, y_fit, color='#222222', lw=1.4, ls='--',
                         label=f'LS 拟合  b={b_ls:.3f}')

            # MLE b 值
            b_m, _ = _mle_b(amps_s, B_AMP_MIN)
            ax3.set_xlabel('振幅 (dB)', fontsize=9)
            ax3.set_ylabel('log10(N≥AMP)', fontsize=9)
            ax3.set_title(f'G-R 分布：{lbl}  ({len(amps_s)} hits)', fontsize=9,
                          pad=3, color=col)
            ax3.legend(fontsize=7, framealpha=0.75)
            ax3.text(0.97, 0.97,
                     f'MLE  b = {b_m:.3f}' if not np.isnan(b_m) else 'MLE: N/A',
                     transform=ax3.transAxes, fontsize=8,
                     ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='white',
                               edgecolor=col, alpha=0.85))
            ax3.grid(True, alpha=0.18, linewidth=0.4)

        out = os.path.join(out_dir, 'step3_b值分析.png')
        fig.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
        return out
    finally:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# § 8  主流程
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Step 3 — AE 事件空间分布 + b 值分析")
    print("=" * 60)

    # 确定使用哪个 clean hits 文件（step2 > step1）
    if os.path.exists(STEP2_AE_CLEAN):
        clean_path = STEP2_AE_CLEAN
        src_label  = 'step2 手动清理'
    elif os.path.exists(STEP1_AE_CLEAN):
        clean_path = STEP1_AE_CLEAN
        src_label  = 'step1 自动滤波'
    else:
        print(f"\n错误: 找不到任何干净 hits 文件，请先运行 step1_auto_filter.py")
        sys.exit(1)

    print(f"[1/6]  读取干净 hits: {src_label}")
    clean_df = pd.read_csv(clean_path)
    print(f"       {len(clean_df)} hits  通道: {sorted(clean_df['CH'].unique())}")

    print(f"[2/6]  解析事件文件: {os.path.basename(AE_EVTS)}")
    if not os.path.exists(AE_EVTS):
        print(f"错误: 找不到 {AE_EVTS}")
        sys.exit(1)
    raw_events = parse_ae_events(AE_EVTS)
    print(f"       共 {len(raw_events)} 个定位事件")

    print(f"[3/6]  构建事件表（q≥{EVT_Q_MIN}，干净占比≥{EVT_CLEAN_FRAC:.0%}）")
    events_df = build_events_df(raw_events, clean_df, EVT_Q_MIN, EVT_CLEAN_FRAC)
    df_clean  = events_df[events_df['is_clean']].reset_index(drop=True)
    n_all     = len(events_df)
    n_clean   = len(df_clean)
    pct       = 100. * n_clean / n_all if n_all > 0 else 0.
    print(f"       总事件 {n_all}  →  保留干净事件 {n_clean} ({pct:.1f}%)")

    os.makedirs(STEP3_DIR, exist_ok=True)

    # 保存事件 CSV
    csv_path  = os.path.join(STEP3_DIR, 'step3_events_clean.csv')
    save_cols = ['Time', 'x', 'y', 'z', 'q', 'src_amp', 'total_abs_e',
                 'n_hits', 'n_clean', 'clean_f']
    df_clean[save_cols].to_csv(csv_path, index=False)
    print(f"       CSV → {csv_path}")

    if n_clean == 0:
        print("警告: 过滤后无可用事件，请检查 EVT_Q_MIN / EVT_CLEAN_FRAC 阈值。")
        sys.exit(0)

    print(f"[4/6]  生成 2D 分布图")
    out2d = gen_2d_figure(df_clean, STEP3_DIR)
    print(f"       → {out2d}")

    print(f"[5/6]  生成 3D 分布图")
    out3d_t, out3d_a = gen_3d_figure(df_clean, STEP3_DIR)
    out3d_e = gen_3d_energy_figure(df_clean, STEP3_DIR)
    out_evo = gen_evolution_figure(df_clean, STEP3_DIR)
    print(f"       → {out3d_t}")
    print(f"       → {out3d_a}")
    print(f"       → {out3d_e}")
    print(f"       → {out_evo}")

    print(f"[6/6]  b 值分析（窗口 {B_WINDOW_N} hits，步长 {B_STEP_N} hits）")
    b_df    = compute_b_series(clean_df)
    out_b   = gen_b_value_figure(b_df, clean_df, STEP3_DIR)
    b_valid = b_df.dropna(subset=['b'])
    print(f"       → {out_b}")
    if len(b_valid):
        print(f"       b 值范围: {b_valid['b'].min():.3f} ~ {b_valid['b'].max():.3f}"
              f"  均值: {b_valid['b'].mean():.3f}")

    # ── 统计摘要 ──────────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  干净事件统计（来源: {src_label}）")
    print(f"{'─'*52}")
    print(f"  总数:  {n_clean}")
    print(f"  时间:  {df_clean['Time'].min():.1f} ~ {df_clean['Time'].max():.1f} s")
    print(f"  X:     {df_clean['x'].min():.1f} ~ {df_clean['x'].max():.1f} mm")
    print(f"  Y:     {df_clean['y'].min():.1f} ~ {df_clean['y'].max():.1f} mm  (高度)")
    print(f"  Z:     {df_clean['z'].min():.1f} ~ {df_clean['z'].max():.1f} mm")
    print(f"  振幅:  {df_clean['src_amp'].min():.0f} ~ {df_clean['src_amp'].max():.0f} dB")
    print(f"  质量:  {df_clean['q'].min():.2f} ~ {df_clean['q'].max():.2f}")
    print(f"{'─'*52}")
    print(f"\n完成。结果目录: {STEP3_DIR}")
