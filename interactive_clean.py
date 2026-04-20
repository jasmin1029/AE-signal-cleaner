#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式AE数据手动清理工具 v2
  多次套索叠加选择 + 滚轮缩放 + 右键拖拽平移

操作:
  点击通道图 / 数字键 1-6  → 切换激活通道
  L  / [切换模式] 按钮     → 在 [套索] / [导航] 模式间切换
  套索模式下左键拖拽        → 圈选点（多次套索叠加）
  滚轮                      → 缩放（任何模式下均有效）
  右键拖拽                  → 平移（任何模式下均有效）
  R  / [重置视图]           → 恢复所有子图到初始范围
  D  / [标记删除]           → 将选中点（橙色）标为删除（灰色）
  U  / [取消标记]           → 将选中点恢复为保留（蓝色）
  Z  / [撤  销]             → 撤销上一步操作（可多次）
  Esc / [清除选择]          → 取消当前橙色选中
  S  / [导出结果]           → 写 CSV + 生成对比图 PNG
"""

import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import os

# ── 交互后端（pyplot import 之前） ────────────────────────────────────────
import matplotlib
_GUI_BACKEND = None
for _b in ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'WXAgg']:
    try:
        matplotlib.use(_b)
        import matplotlib.pyplot as _pt; _f = _pt.figure(); _pt.close(_f)
        _GUI_BACKEND = _b; break
    except Exception:
        pass
if _GUI_BACKEND is None:
    matplotlib.use('TkAgg')
    _GUI_BACKEND = 'TkAgg (fallback)'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import LassoSelector, Button
from matplotlib.path import Path
from matplotlib.ticker import MultipleLocator

for _fn in ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(_fn, fallback_to_default=False)
        plt.rcParams['font.family'] = _fn; break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid']          = True
plt.rcParams['grid.alpha']         = 0.20
plt.rcParams['grid.linewidth']     = 0.4

# ── 路径 ──────────────────────────────────────────────────────────────────
BASE       = r'g:\Cursor project\ZCY-shengfashe'
RESULT_DIR = os.path.join(BASE, '结果')
AE_CLEAN   = os.path.join(RESULT_DIR, 'v17_AE_clean.csv')
AE_RAW_TXT = os.path.join(BASE, '声发射', '04-15-hits-振铃计数、能量等.TXT')

CH_COLORS  = {1:'#1f77b4', 2:'#ff7f0e', 3:'#2ca02c',
              4:'#d62728', 5:'#9467bd', 6:'#8c564b'}
MODE_SEL   = 'select'
MODE_NAV   = 'navigate'


def parse_ae_hits(path: str) -> pd.DataFrame:
    """解析仪器原始 AE hits 文本文件。"""
    rows = []
    with open(path, 'r', errors='replace') as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) >= 9:
                try:
                    rows.append({
                        'Time':     float(p[1]),
                        'CH':       int(p[2]),
                        'RISE':     int(p[3]),
                        'COUN':     int(p[4]),
                        'ENER':     int(p[5]),
                        'DURATION': int(p[6]),
                        'AMP':      float(p[7]),
                        'ABS_E':    float(p[8]),
                    })
                except (ValueError, IndexError):
                    pass
    df = pd.DataFrame(rows)
    if len(df):
        df = df[df['Time'] > 0].sort_values('Time').reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════
class ManualCleaner:
    C_KEEP = '#4488cc'
    C_DEL  = '#cccccc'
    C_SEL  = '#ff8800'
    C_ABG  = '#fffbee'   # 激活通道背景

    def __init__(self, data: pd.DataFrame):
        self.data   = data.reset_index(drop=True)
        self.n      = len(self.data)
        self.t_arr  = self.data['Time'].values.astype(float)
        self.a_arr  = self.data['AMP'].values.astype(float)
        self.ch_arr = self.data['CH'].values.astype(int)

        self.keep     = np.ones(self.n,  dtype=bool)
        self.selected = np.zeros(self.n, dtype=bool)
        self.history  = []                 # undo stack

        self.ch_idx   = {ch: np.where(self.ch_arr == ch)[0]
                         for ch in range(1, 7)}
        self.active_ch = 1
        self.mode      = MODE_SEL
        self.lasso     = None
        self._pan_data = None              # right-click pan state

        self._build_ui()
        self._set_mode(MODE_SEL, init=True)
        self._activate(1)

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.fig = plt.figure(figsize=(22, 13))
        try:
            self.fig.canvas.manager.set_window_title('AE 手动清理工具 v2')
        except Exception:
            pass

        self.fig.text(
            0.5, 0.975,
            '1-6 切换通道  │  L 切换模式  │  套索(左键) 叠加选取  │  '
            '滚轮缩放  右键平移  R 重置  │  D删  U恢复  Z撤销  S导出',
            ha='center', fontsize=9, color='#444'
        )

        # 6 个通道子图
        gs = gridspec.GridSpec(3, 2, figure=self.fig,
                               hspace=0.44, wspace=0.20,
                               top=0.93, bottom=0.13,
                               left=0.06, right=0.97)
        pos = {1:(0,0), 2:(0,1), 3:(1,0), 4:(1,1), 5:(2,0), 6:(2,1)}

        self._xlo = float(self.t_arr.min()) - 5
        self._xhi = float(self.t_arr.max()) + 15
        self._ylo = float(self.a_arr.min()) - 3
        self._yhi = float(self.a_arr.max()) + 3

        self.axes     = {}
        self.scatters = {}
        for ch in range(1, 7):
            r, c = pos[ch]
            ax = self.fig.add_subplot(gs[r, c])
            self.axes[ch] = ax
            idx = self.ch_idx[ch]
            sc = ax.scatter(
                self.t_arr[idx], self.a_arr[idx],
                s=1.5, alpha=0.4,
                facecolors=[self.C_KEEP] * len(idx),
                edgecolors='none', zorder=2, rasterized=False
            )
            self.scatters[ch] = sc
            ax.set_xlim(self._xlo, self._xhi)
            ax.set_ylim(self._ylo, self._yhi)
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            ax.set_xlabel('时间 (s)', fontsize=8)
            ax.set_ylabel('振幅 (dB)', fontsize=8)
            ax.tick_params(labelsize=8)

        # 按钮行
        bw, bh, by, gap = 0.108, 0.046, 0.025, 0.007
        bx0 = 0.050
        defs = [
            ('[套索]✓ / [导航]',  '#ddf4ff', '#88ccff', self._cb_toggle_mode),
            ('重置视图  [R]',      '#eeeeff', '#aaaaee', self._cb_reset),
            ('标记删除  [D]',      '#ffddcc', '#ff9966', self._cb_mark_delete),
            ('取消标记  [U]',      '#ccddff', '#6699ee', self._cb_unmark),
            ('撤  销   [Z]',       '#eeeedd', '#cccc77', self._cb_undo),
            ('清除选择 [Esc]',     '#eeeeee', '#bbbbbb', self._cb_clear_sel),
            ('导出结果  [S]',      '#ccffcc', '#44bb44', self._cb_export),
        ]
        self.btn_objs = []
        for i, (lbl, fc, hc, cb) in enumerate(defs):
            ax_b = self.fig.add_axes([bx0 + i*(bw+gap), by, bw, bh])
            btn  = Button(ax_b, lbl, color=fc, hovercolor=hc)
            btn.on_clicked(cb)
            self.btn_objs.append(btn)
        self._btn_mode = self.btn_objs[0]

        # 状态栏
        ax_st = self.fig.add_axes([0.870, by, 0.122, bh])
        ax_st.axis('off')
        self._status_txt = ax_st.text(
            0.5, 0.5, '', ha='center', va='center', fontsize=8,
            transform=ax_st.transAxes,
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='#fffff0', edgecolor='#aaaaaa', alpha=0.95)
        )

        # 事件绑定
        self.fig.canvas.mpl_connect('key_press_event',      self._on_key)
        self.fig.canvas.mpl_connect('button_press_event',   self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_move)
        self.fig.canvas.mpl_connect('scroll_event',         self._on_scroll)

    # ── 模式切换 ─────────────────────────────────────────────────────────
    def _set_mode(self, mode, init=False):
        self.mode = mode
        # 断开旧套索
        if self.lasso is not None:
            try: self.lasso.disconnect_events()
            except Exception: pass
            self.lasso = None
        if mode == MODE_SEL:
            # 连接到当前激活通道
            if not init:
                self.lasso = LassoSelector(
                    self.axes[self.active_ch], self._on_lasso,
                    useblit=True, button=[1]
                )
            self._btn_mode.label.set_text('[套索]✓ / [导航]')
            self._btn_mode.ax.set_facecolor('#ddf4ff')
        else:
            self._btn_mode.label.set_text('[套索] / [导航]✓')
            self._btn_mode.ax.set_facecolor('#aaccff')
        self._update_status()
        if not init:
            self.fig.canvas.draw_idle()

    # ── 通道激活 ─────────────────────────────────────────────────────────
    def _activate(self, ch):
        old = self.active_ch
        # 恢复旧通道
        if old in self.axes and old != ch:
            self.axes[old].set_facecolor('white')
            self._update_title(old)
        # 断开旧套索
        if self.lasso is not None:
            try: self.lasso.disconnect_events()
            except Exception: pass
            self.lasso = None
        # 清选中
        self.selected[:] = False
        self.active_ch = ch
        self.axes[ch].set_facecolor(self.C_ABG)
        # 建套索（仅选择模式）
        if self.mode == MODE_SEL:
            self.lasso = LassoSelector(
                self.axes[ch], self._on_lasso,
                useblit=True, button=[1]
            )
        self._update_title(ch)
        self._refresh_colors(ch)
        self._update_status()
        self.fig.canvas.draw_idle()

    # ── 套索回调（多次叠加） ─────────────────────────────────────────────
    def _on_lasso(self, verts):
        if len(verts) < 3:
            return
        ch  = self.active_ch
        idx = self.ch_idx[ch]
        if len(idx) == 0:
            return
        xy   = np.column_stack([self.t_arr[idx], self.a_arr[idx]])
        path = Path(verts)
        try:
            inside = path.contains_points(xy)
        except Exception:
            return
        # ★ 叠加：新选区 OR 进已有选中
        for li, gi in enumerate(idx):
            if inside[li] and self.keep[gi]:
                self.selected[gi] = True        # 不 clear，叠加
        self._refresh_colors(ch)
        self._update_status()
        self.fig.canvas.draw_idle()

    # ── 滚轮缩放 ─────────────────────────────────────────────────────────
    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None or ax not in self.axes.values():
            return
        factor = 0.82 if event.button == 'up' else 1.0/0.82
        xd, yd = event.xdata, event.ydata
        if xd is None or yd is None:
            return
        xl, xh = ax.get_xlim()
        yl, yh = ax.get_ylim()
        ax.set_xlim(xd + (xl-xd)*factor, xd + (xh-xd)*factor)
        ax.set_ylim(yd + (yl-yd)*factor, yd + (yh-yd)*factor)
        self.fig.canvas.draw_idle()

    # ── 右键平移 ─────────────────────────────────────────────────────────
    def _on_press(self, event):
        if event.button == 3 and event.inaxes in self.axes.values():
            ax = event.inaxes
            xl, xh = ax.get_xlim()
            yl, yh = ax.get_ylim()
            # 记录初始状态（像素坐标 + 数据范围 + 当时的比例）
            ax_pos = ax.get_position()
            fig_px = self.fig.get_size_inches() * self.fig.dpi
            ax_w_px = ax_pos.width  * fig_px[0]
            ax_h_px = ax_pos.height * fig_px[1]
            self._pan_data = dict(
                x_px=event.x, y_px=event.y,
                ax=ax,
                xlim=(xl, xh), ylim=(yl, yh),
                x_scale=(xh-xl)/ax_w_px if ax_w_px > 0 else 1,
                y_scale=(yh-yl)/ax_h_px if ax_h_px > 0 else 1,
            )
            return
        # 左键点击非激活通道 → 切换激活
        if event.button == 1:
            for ch, ax in self.axes.items():
                if event.inaxes == ax and ch != self.active_ch:
                    self._activate(ch)
                    return

    def _on_release(self, event):
        if event.button == 3:
            self._pan_data = None

    def _on_move(self, event):
        if self._pan_data is None:
            return
        p  = self._pan_data
        dx = (event.x - p['x_px']) * p['x_scale']
        dy = (event.y - p['y_px']) * p['y_scale']
        xl0, xh0 = p['xlim']
        yl0, yh0 = p['ylim']
        p['ax'].set_xlim(xl0-dx, xh0-dx)
        p['ax'].set_ylim(yl0-dy, yh0-dy)
        self.fig.canvas.draw_idle()

    # ── 操作按钮/键盘 ────────────────────────────────────────────────────
    def _cb_toggle_mode(self, event=None):
        self._set_mode(MODE_NAV if self.mode == MODE_SEL else MODE_SEL)

    def _cb_reset(self, event=None):
        for ax in self.axes.values():
            ax.set_xlim(self._xlo, self._xhi)
            ax.set_ylim(self._ylo, self._yhi)
        self.fig.canvas.draw_idle()

    def _cb_mark_delete(self, event=None):
        if not self.selected.any():
            return
        self.history.append(self.keep.copy())
        self.keep[self.selected] = False
        self.selected[:] = False
        self._refresh_colors(self.active_ch)
        self._update_title(self.active_ch)
        self._update_status()
        self.fig.canvas.draw_idle()

    def _cb_unmark(self, event=None):
        if not self.selected.any():
            return
        self.history.append(self.keep.copy())
        self.keep[self.selected] = True
        self.selected[:] = False
        self._refresh_colors(self.active_ch)
        self._update_title(self.active_ch)
        self._update_status()
        self.fig.canvas.draw_idle()

    def _cb_undo(self, event=None):
        if not self.history:
            return
        self.keep     = self.history.pop()
        self.selected[:] = False
        self._refresh_colors(self.active_ch)
        self._update_title(self.active_ch)
        self._update_status()
        self.fig.canvas.draw_idle()

    def _cb_clear_sel(self, event=None):
        self.selected[:] = False
        self._refresh_colors(self.active_ch)
        self._update_status()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        k = event.key
        if   k == 'd':      self._cb_mark_delete()
        elif k == 'u':      self._cb_unmark()
        elif k == 'z':      self._cb_undo()
        elif k == 's':      self._cb_export()
        elif k == 'l':      self._cb_toggle_mode()
        elif k == 'r':      self._cb_reset()
        elif k == 'escape': self._cb_clear_sel()
        elif k in [str(i) for i in range(1, 7)]:
            self._activate(int(k))

    # ── 刷新显示 ─────────────────────────────────────────────────────────
    def _refresh_colors(self, ch):
        idx = self.ch_idx[ch]
        colors = []
        for gi in idx:
            if   self.selected[gi]:  colors.append(self.C_SEL)
            elif not self.keep[gi]:  colors.append(self.C_DEL)
            else:                    colors.append(self.C_KEEP)
        self.scatters[ch].set_facecolors(colors)

    def _update_title(self, ch):
        idx  = self.ch_idx[ch]
        kept = int(self.keep[idx].sum())
        deld = int((~self.keep[idx]).sum())
        sel  = int(self.selected[idx].sum())
        mode = f' [{self.mode[:3].upper()}]' if ch == self.active_ch else ''
        mark = '  ◀' if ch == self.active_ch else ''
        self.axes[ch].set_title(
            f'CH{ch}  保留 {kept}  已删 {deld}  选中 {sel}{mode}{mark}',
            fontsize=8, pad=2
        )

    def _update_status(self):
        n_k = int(self.keep.sum())
        n_d = int((~self.keep).sum())
        n_s = int(self.selected.sum())
        md  = '套索' if self.mode == MODE_SEL else '导航'
        self._status_txt.set_text(
            f'模式: {md}\n保留 {n_k} | 删 {n_d}\n选中 {n_s}'
        )

    # ── 导出（外层捕获异常并通知用户） ──────────────────────────────────────
    def _cb_export(self, event=None):
        try:
            self._do_export()
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(f"\n[导出错误]\n{tb}")
            self._status_txt.set_text('导出失败!\n见终端')
            self.fig.canvas.draw_idle()
            # 也弹出 Tk 提示框
            try:
                import tkinter.messagebox as mb
                mb.showerror('导出失败', tb[-300:])
            except Exception:
                pass

    def _do_export(self):
        os.makedirs(RESULT_DIR, exist_ok=True)
        ae_final   = self.data[self.keep].reset_index(drop=True)
        ae_removed = self.data[~self.keep].reset_index(drop=True)

        out_clean = os.path.join(RESULT_DIR, 'v17_manual_AE_clean.csv')
        out_del   = os.path.join(RESULT_DIR, 'v17_manual_AE_deleted.csv')
        ae_final.to_csv(out_clean, index=False)
        ae_removed.to_csv(out_del, index=False)

        n_f = len(ae_final); n_d = len(ae_removed)
        pct = 100.*n_d/self.n if self.n > 0 else 0.
        print(f"\n[导出]  保留 {n_f}  手动删除 {n_d} ({pct:.1f}%)")
        print(f"  {out_clean}")
        print(f"  {out_del}")

        self._status_txt.set_text(f'生成图中...')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        out_png  = self._gen_figure(ae_final, ae_removed)
        out_cmp  = self._gen_comparison_figure(ae_final)
        print(f"  统计图: {out_png}")
        print(f"  对比图: {out_cmp}")

        print("\n  CH    总计   保留   删除   删除率")
        for ch in range(1, 7):
            orig = len(self.ch_idx[ch])
            kept = int((ae_final['CH'] == ch).sum())
            deld = int((ae_removed['CH'] == ch).sum())
            r    = 100.*deld/orig if orig > 0 else 0.
            print(f"  CH{ch}  {orig:>5}  {kept:>5}  {deld:>5}  {r:>6.1f}%")

        self._status_txt.set_text(f'已导出!\n保留 {n_f}\n删除 {n_d}')
        self.fig.canvas.draw_idle()

        # Tk 成功提示
        try:
            import tkinter.messagebox as mb
            mb.showinfo('导出完成',
                        f'保留 {n_f} hits，手动删除 {n_d} hits\n\n'
                        f'CSV: {os.path.basename(out_clean)}\n'
                        f'图1: {os.path.basename(out_png)}\n'
                        f'图2: {os.path.basename(out_cmp)}\n\n'
                        f'目录: {RESULT_DIR}')
        except Exception:
            pass

    # ── 生成静态输出图（plt 非交互模式，保存后立即关闭） ─────────────────
    def _gen_figure(self, ae_final: pd.DataFrame,
                    ae_removed: pd.DataFrame) -> str:
        # ── 数据准备 ──────────────────────────────────────────────────
        ae_before = pd.concat([ae_final, ae_removed]).sort_values('Time').reset_index(drop=True)
        ae_after  = ae_final.sort_values('Time').reset_index(drop=True)

        xmax = float(ae_before['Time'].max()) + 15
        ylo  = float(ae_before['AMP'].min()) - 3
        yhi  = float(ae_before['AMP'].max()) + 3
        n_f  = len(ae_final); n_d = len(ae_removed)
        pct  = 100.*n_d/(n_f+n_d) if (n_f+n_d) > 0 else 0.
        SZ, AL = 1.5, 0.40

        # 累计振铃计数 & 累计绝对能量（手动清理前后对比）
        cum_coun_before = ae_before['COUN'].cumsum().values
        cum_ener_before = ae_before['ABS_E'].cumsum().values
        cum_coun_after  = ae_after['COUN'].cumsum().values
        cum_ener_after  = ae_after['ABS_E'].cumsum().values

        # 各通道累计振铃计数（清理后）
        ch_coun = {}; ch_ener = {}
        for ch in range(1, 7):
            sub = ae_after[ae_after['CH'] == ch].sort_values('Time')
            ch_coun[ch] = (sub['Time'].values, sub['COUN'].cumsum().values)
            ch_ener[ch] = (sub['Time'].values, sub['ABS_E'].cumsum().values)

        # ── 绘图（8行×2列：6行散点 + 2行累计曲线）───────────────────
        plt.ioff()
        fig2 = plt.figure(figsize=(20, 36))
        try:
            fig2.suptitle(
                f'声发射 手动清理后最终结果  (v17 + 手动)\n'
                f'保留 {n_f} / {n_f+n_d}    手动删除 {n_d} ({pct:.1f}%)',
                fontsize=11, fontweight='bold'
            )
            gs = gridspec.GridSpec(
                8, 2, figure=fig2,
                height_ratios=[1, 1, 1, 1, 1, 1, 1.4, 1.4],
                hspace=0.28, wspace=0.06,
                top=0.94, bottom=0.03, left=0.07, right=0.97
            )

            # ── 6 通道散点图 ──────────────────────────────────────
            for ch in range(1, 7):
                row = ch - 1
                d_d = ae_removed[ae_removed['CH'] == ch] if n_d > 0 else ae_removed
                d_k = ae_final[ae_final['CH'] == ch]

                axL = fig2.add_subplot(gs[row, 0])
                if len(d_d):
                    axL.scatter(d_d['Time'], d_d['AMP'],
                                s=SZ, alpha=AL*0.35, color='silver',
                                zorder=1, rasterized=True)
                if len(d_k):
                    axL.scatter(d_k['Time'], d_k['AMP'],
                                s=SZ, alpha=AL, color=CH_COLORS[ch],
                                zorder=2, rasterized=True)
                axL.set_ylabel(f'CH{ch}\n振幅(dB)', fontsize=9)
                axL.set_ylim(ylo, yhi); axL.set_xlim(0, xmax)
                axL.yaxis.set_major_locator(MultipleLocator(20))
                axL.yaxis.set_minor_locator(MultipleLocator(10))
                axL.grid(True, alpha=0.2, linewidth=0.4)
                axL.text(0.01, 0.97,
                         f'保留 {len(d_k)}  删 {len(d_d)}',
                         transform=axL.transAxes, fontsize=7,
                         va='top', color='#333')
                if row == 0:
                    axL.set_title('原始（灰=已删  彩=保留）', fontsize=10, pad=6)

                axR = fig2.add_subplot(gs[row, 1], sharey=axL)
                if len(d_k):
                    axR.scatter(d_k['Time'], d_k['AMP'],
                                s=SZ, alpha=AL, color=CH_COLORS[ch],
                                rasterized=True)
                axR.set_xlim(0, xmax); axR.set_ylim(ylo, yhi)
                axR.yaxis.set_major_locator(MultipleLocator(20))
                axR.tick_params(labelleft=False)
                axR.grid(True, alpha=0.2, linewidth=0.4)
                if row == 0:
                    axR.set_title(f'最终保留 ({n_f} hits)', fontsize=10, pad=6)
                if row < 5:
                    axL.tick_params(labelbottom=False)
                    axR.tick_params(labelbottom=False)
                else:
                    axL.set_xlabel('时间 (s)')
                    axR.set_xlabel('时间 (s)')

            # ── 累计振铃计数（row 6，跨两列）────────────────────────
            ax_c = fig2.add_subplot(gs[6, :])
            # 手动清理前（灰色背景参考线）
            ax_c.plot(ae_before['Time'].values, cum_coun_before,
                      color='#bbbbbb', lw=1.2, ls='--', zorder=1,
                      label=f'手动清理前 ({len(ae_before)} hits)')
            # 各通道分色（清理后）
            for ch in range(1, 7):
                t_ch, c_ch = ch_coun[ch]
                if len(t_ch):
                    ax_c.plot(t_ch, c_ch, color=CH_COLORS[ch],
                              lw=0.8, alpha=0.7, zorder=2,
                              label=f'CH{ch}')
            # 总计（清理后，黑色加粗）
            ax_c.plot(ae_after['Time'].values, cum_coun_after,
                      color='black', lw=1.6, zorder=3,
                      label=f'清理后合计 ({n_f} hits)')
            ax_c.set_xlim(0, xmax)
            ax_c.set_ylabel('累计振铃计数', fontsize=10)
            ax_c.set_xlabel('时间 (s)', fontsize=9)
            ax_c.set_title('累计振铃计数  (灰虚=清理前，彩=各通道，黑=清理后合计)',
                           fontsize=10, pad=4)
            ax_c.legend(ncol=8, fontsize=7, loc='upper left',
                        framealpha=0.7)
            ax_c.grid(True, alpha=0.2, linewidth=0.4)
            ax_c.yaxis.get_major_formatter().set_scientific(False)

            # ── 累计绝对能量（row 7，跨两列，对数纵轴）──────────────
            ax_e = fig2.add_subplot(gs[7, :])
            ax_e.plot(ae_before['Time'].values, cum_ener_before,
                      color='#bbbbbb', lw=1.2, ls='--', zorder=1,
                      label=f'手动清理前')
            for ch in range(1, 7):
                t_ch, e_ch = ch_ener[ch]
                if len(t_ch) and e_ch[-1] > 0:
                    ax_e.plot(t_ch, e_ch, color=CH_COLORS[ch],
                              lw=0.8, alpha=0.7, zorder=2,
                              label=f'CH{ch}')
            ax_e.plot(ae_after['Time'].values, cum_ener_after,
                      color='black', lw=1.6, zorder=3,
                      label=f'清理后合计')
            ax_e.set_xlim(0, xmax)
            ax_e.set_yscale('log')
            ax_e.set_ylabel('累计绝对能量 (aJ)', fontsize=10)
            ax_e.set_xlabel('时间 (s)', fontsize=9)
            ax_e.set_title('累计绝对能量  对数纵轴  (灰虚=清理前，彩=各通道，黑=清理后合计)',
                           fontsize=10, pad=4)
            ax_e.legend(ncol=8, fontsize=7, loc='upper left',
                        framealpha=0.7)
            ax_e.grid(True, alpha=0.2, linewidth=0.4, which='both')

            out_png = os.path.join(RESULT_DIR, 'v17_manual_清理结果.png')
            fig2.savefig(out_png, dpi=200, bbox_inches='tight')
            return out_png
        finally:
            plt.close(fig2)
            plt.ion()   # 恢复交互模式

    # ── 对比图：仪器原始信号 vs 最终结果（每通道左右并排）────────────────
    def _gen_comparison_figure(self, ae_final: pd.DataFrame) -> str:
        """
        左列：仪器原始未去干扰（直接读 AE_RAW_TXT）
        右列：最终保留（v17 + 手动清理后）
        """
        # 直接读仪器原始文件
        if os.path.exists(AE_RAW_TXT):
            print(f"  读取原始文件: {AE_RAW_TXT}")
            ae_raw = parse_ae_hits(AE_RAW_TXT)
            raw_label = f'仪器原始信号 ({len(ae_raw)} hits)'
        else:
            # 退而求其次：用 v17 两个 CSV 拼合
            print(f"  警告: 找不到原始TXT，改用v17 CSV拼合")
            v17_contam_path = os.path.join(RESULT_DIR, 'v17_AE_contaminated.csv')
            if os.path.exists(v17_contam_path):
                v17_contam = pd.read_csv(v17_contam_path)
                ae_raw = pd.concat([self.data, v17_contam]).sort_values('Time').reset_index(drop=True)
            else:
                ae_raw = self.data.sort_values('Time').reset_index(drop=True)
            raw_label = f'原始信号（CSV拼合，{len(ae_raw)} hits）'

        ae_clean = ae_final.sort_values('Time').reset_index(drop=True)
        n_raw = len(ae_raw); n_clean = len(ae_clean)
        removed_total = n_raw - n_clean
        pct = 100. * removed_total / n_raw if n_raw > 0 else 0.

        xmax = float(ae_raw['Time'].max()) + 15
        ylo  = float(ae_raw['AMP'].min()) - 3
        yhi  = float(ae_raw['AMP'].max()) + 3
        SZ, AL = 1.5, 0.35

        plt.ioff()
        fig3 = plt.figure(figsize=(20, 28))
        try:
            fig3.suptitle(
                f'声发射信号  未去干扰 vs 最终结果\n'
                f'原始 {n_raw} hits  →  最终 {n_clean} hits'
                f'  (共去除 {removed_total} = {pct:.1f}%)',
                fontsize=11, fontweight='bold'
            )
            gs = gridspec.GridSpec(
                6, 2, figure=fig3,
                hspace=0.10, wspace=0.10,
                top=0.93, bottom=0.05, left=0.07, right=0.97
            )

            for ch in range(1, 7):
                row = ch - 1
                raw_ch   = ae_raw[ae_raw['CH'] == ch]
                clean_ch = ae_clean[ae_clean['CH'] == ch]

                # 左：原始未去干扰
                axL = fig3.add_subplot(gs[row, 0])
                axL.scatter(raw_ch['Time'], raw_ch['AMP'],
                            s=SZ, alpha=AL, color=CH_COLORS[ch],
                            zorder=2, rasterized=True)
                axL.set_xlim(0, xmax); axL.set_ylim(ylo, yhi)
                axL.yaxis.set_major_locator(MultipleLocator(20))
                axL.yaxis.set_minor_locator(MultipleLocator(10))
                axL.set_ylabel(f'CH{ch}\n振幅 (dB)', fontsize=9)
                axL.grid(True, alpha=0.2, linewidth=0.4)
                axL.text(0.01, 0.97, f'{len(raw_ch)} hits',
                         transform=axL.transAxes, fontsize=7,
                         va='top', color='#333')
                if row == 0:
                    axL.set_title(raw_label, fontsize=10, pad=6)

                # 右：最终保留
                axR = fig3.add_subplot(gs[row, 1], sharey=axL)
                axR.scatter(clean_ch['Time'], clean_ch['AMP'],
                            s=SZ, alpha=AL, color=CH_COLORS[ch],
                            zorder=2, rasterized=True)
                axR.set_xlim(0, xmax); axR.set_ylim(ylo, yhi)
                axR.yaxis.set_major_locator(MultipleLocator(20))
                axR.tick_params(labelleft=False)
                axR.grid(True, alpha=0.2, linewidth=0.4)
                axR.text(0.01, 0.97,
                         f'{len(clean_ch)} hits  (去除 {len(raw_ch)-len(clean_ch)},'
                         f' {100.*(len(raw_ch)-len(clean_ch))/len(raw_ch):.1f}%)'
                         if len(raw_ch) > 0 else f'{len(clean_ch)} hits',
                         transform=axR.transAxes, fontsize=7,
                         va='top', color='#333')
                if row == 0:
                    axR.set_title(f'最终保留 ({n_clean} hits)', fontsize=10, pad=6)

                if row < 5:
                    axL.tick_params(labelbottom=False)
                    axR.tick_params(labelbottom=False)
                else:
                    axL.set_xlabel('时间 (s)')
                    axR.set_xlabel('时间 (s)')

            out_cmp = os.path.join(RESULT_DIR, 'v17_manual_原始vs最终对比.png')
            fig3.savefig(out_cmp, dpi=200, bbox_inches='tight')
            return out_cmp
        finally:
            plt.close(fig3)
            plt.ion()


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("="*60)
    print(f"交互式AE手动清理工具 v2  (后端: {_GUI_BACKEND})")

    if not os.path.exists(AE_CLEAN):
        print(f"\n错误: 找不到 {AE_CLEAN}")
        print("请先运行 analysis_v17.py")
        sys.exit(1)

    data = pd.read_csv(AE_CLEAN)
    print(f"加载 {len(data)} hits  通道: {sorted(data['CH'].unique())}\n")
    print("快捷键:")
    print("  1-6      切换激活通道")
    print("  L        [套索选择] / [导航] 模式切换")
    print("  左键拖   套索圈点（多次叠加，套索模式下）")
    print("  滚轮     缩放（始终有效）")
    print("  右键拖   平移（始终有效）")
    print("  R        重置所有子图视图")
    print("  D        标记选中点为删除（灰色）")
    print("  U        恢复选中点为保留（蓝色）")
    print("  Z        撤销上一步")
    print("  Esc      清除当前选中（不删除）")
    print("  S        导出 CSV + 生成统计图")

    cleaner = ManualCleaner(data)
    plt.show()
