#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ae_cleaner_web.py — 声发射信号手动清理工具（网页版）
=====================================================
使用方法:
  pip install dash plotly pandas numpy
  python ae_cleaner_web.py
  然后在浏览器访问 http://127.0.0.1:8050

功能:
  • 上传仪器原始 AE hits TXT 文件（格式与 Express-8 / PAC 兼容）
  • 6 通道振幅-时间散点图，支持套索 / 框选 / 缩放 / 平移
  • 跨通道多次选择后批量标记删除（灰色）或恢复保留
  • 多步撤销（最多 30 步）
  • 一键导出干净数据 CSV
  • 键盘快捷键：D 删除 / U 恢复 / Z 撤销

依赖:
  pip install "dash>=2.9" plotly pandas numpy
"""

# ── 依赖检查 ────────────────────────────────────────────────────────────────
try:
    import dash
except ImportError:
    print("\n错误: 缺少 dash 库")
    print("请运行: pip install \"dash>=2.9\" plotly pandas numpy\n")
    raise

import base64
import io

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# ── 颜色配置 ────────────────────────────────────────────────────────────────
CH_COLORS = {
    1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c',
    4: '#d62728', 5: '#9467bd', 6: '#8c564b',
}
C_DEL = '#bbbbbb'   # 已删除点
MAX_UNDO = 30       # 最大撤销步数


# ═══════════════════════════════════════════════════════════════════════════
# 数据解析
# ═══════════════════════════════════════════════════════════════════════════
def parse_ae_hits(text: str) -> pd.DataFrame:
    """
    解析仪器 AE hits 文本（Express-8 / PAC 格式）。
    列顺序: ID  TIME  CH  RISE  COUN  ENER  DURATION  AMP  ABS_E
    """
    rows = []
    for line in text.splitlines():
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
# 构建图形
# ═══════════════════════════════════════════════════════════════════════════
def build_figures(data_store: dict, keep: list) -> list:
    """
    根据当前 keep 状态为 6 个通道各生成一个 Plotly 散点图。
    删除点 = 灰色，保留点 = 通道色。
    customdata 存储全局索引，供选择回调使用。
    """
    t_arr  = np.array(data_store['Time'],  dtype=float)
    a_arr  = np.array(data_store['AMP'],   dtype=float)
    ch_arr = np.array(data_store['CH'],    dtype=int)
    keep_arr = np.array(keep, dtype=bool)
    n_total  = len(t_arr)

    # 使用数据长度作为 uirevision 基础：新文件加载时重置视图，编辑时保持视图
    ui_base = str(n_total)

    x_min = max(0.0, float(t_arr.min()) - 5)
    x_max = float(t_arr.max()) + 15
    y_min = float(a_arr.min()) - 3
    y_max = float(a_arr.max()) + 3

    figs = []
    for ch in range(1, 7):
        mask  = ch_arr == ch
        idx   = np.where(mask)[0]

        if len(idx) == 0:
            # 空通道
            fig = go.Figure()
            fig.update_layout(
                title=dict(text=f'CH{ch}  (无数据)', x=0.5, font=dict(size=12)),
                height=220, margin=dict(l=48, r=8, t=32, b=36),
                plot_bgcolor='white', paper_bgcolor='white',
            )
            figs.append(fig)
            continue

        t_ch = t_arr[idx]
        a_ch = a_arr[idx]
        colors = [CH_COLORS[ch] if keep_arr[gi] else C_DEL for gi in idx]
        n_kept = int(keep_arr[idx].sum())
        n_del  = int((~keep_arr[idx]).sum())

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=t_ch.tolist(),
            y=a_ch.tolist(),
            mode='markers',
            marker=dict(
                color=colors,
                size=3,
                opacity=0.55,
                line=dict(width=0),
            ),
            customdata=idx.tolist(),   # 全局索引，用于选择映射
            hovertemplate=(
                '时间: %{x:.3f} s<br>'
                '振幅: %{y:.1f} dB<br>'
                '索引: %{customdata}<extra></extra>'
            ),
            showlegend=False,
        ))
        fig.update_layout(
            title=dict(
                text=f'<b>CH{ch}</b>　保留 {n_kept}　已删 {n_del}',
                x=0.5, font=dict(size=11),
            ),
            xaxis=dict(
                title='时间 (s)',
                range=[x_min, x_max],
                gridcolor='#eeeeee',
                zeroline=False,
            ),
            yaxis=dict(
                title='振幅 (dB)',
                range=[y_min, y_max],
                dtick=20,
                gridcolor='#eeeeee',
                zeroline=False,
            ),
            margin=dict(l=48, r=8, t=32, b=36),
            plot_bgcolor='white',
            paper_bgcolor='white',
            dragmode='lasso',
            uirevision=f'ch{ch}-{ui_base}',   # 编辑时保持缩放
            height=220,
            font=dict(size=10, family='Microsoft YaHei, SimHei, Arial'),
            hoverlabel=dict(bgcolor='white', font_size=11),
            newselection=dict(line=dict(color='#ff6600', width=1.5)),
        )
        figs.append(fig)

    return figs


# ═══════════════════════════════════════════════════════════════════════════
# Dash 应用布局
# ═══════════════════════════════════════════════════════════════════════════
_BTN = lambda label, bid, color: html.Button(
    label, id=bid, n_clicks=0,
    style={
        'background': color, 'color': 'white', 'border': 'none',
        'borderRadius': '5px', 'padding': '7px 14px',
        'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': 'bold',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
    }
)

app = Dash(__name__, title='AE 手动清理工具')
server = app.server   # 暴露 Flask server，供 gunicorn 调用
app.layout = html.Div([

    # ── 顶栏 ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H2('声发射信号手动清理工具',
                    style={'margin': 0, 'fontSize': '18px', 'color': '#1a237e'}),
            html.Span('网页版 — 上传 TXT · 套索选点 · 标记删除 · 导出 CSV',
                      style={'fontSize': '11px', 'color': '#555', 'marginLeft': '12px'}),
        ], style={'display': 'flex', 'alignItems': 'baseline'}),
    ], style={'padding': '10px 16px', 'background': '#e8eaf6',
              'borderBottom': '2px solid #3949AB'}),

    # ── 工具栏 ────────────────────────────────────────────────────────────
    html.Div([
        # 上传
        dcc.Upload(
            id='upload-file',
            children=html.Button(
                '📂  上传 AE hits TXT',
                style={'background': '#3949AB', 'color': 'white', 'border': 'none',
                       'borderRadius': '5px', 'padding': '7px 14px',
                       'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': 'bold'}
            ),
            accept='.txt,.TXT',
            style={'display': 'inline-block'},
        ),
        html.Span(id='file-label',
                  style={'marginLeft': '10px', 'color': '#333',
                         'fontSize': '12px', 'alignSelf': 'center'}),

        # 操作按钮（右对齐）
        html.Div([
            _BTN('✂  标记删除  [D]', 'btn-delete', '#c62828'),
            _BTN('✔  恢复保留  [U]', 'btn-unmark',  '#1565C0'),
            _BTN('↩  撤  销   [Z]', 'btn-undo',   '#E65100'),
            _BTN('♻  全部恢复', 'btn-reset',  '#558B2F'),
            _BTN('💾  导出 CSV', 'btn-export', '#00695C'),
            dcc.Download(id='download-csv'),
        ], style={'marginLeft': 'auto', 'display': 'flex',
                  'gap': '8px', 'alignItems': 'center', 'flexWrap': 'wrap'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'padding': '8px 16px',
              'background': '#fafafa', 'borderBottom': '1px solid #ccc',
              'flexWrap': 'wrap', 'gap': '8px'}),

    # ── 状态栏 ────────────────────────────────────────────────────────────
    html.Div(id='status-bar',
             children='请先上传声发射 hits TXT 文件',
             style={'padding': '5px 16px', 'background': '#f1f8e9',
                    'fontSize': '12px', 'color': '#33691E',
                    'borderBottom': '1px solid #dcedc8', 'minHeight': '26px'}),

    # ── 操作提示 ──────────────────────────────────────────────────────────
    html.Div([
        '💡 ',
        html.B('套索 / 框选：'), '在图上拖拽选择点　',
        html.B('右上角工具栏：'), '可切换矩形框选 / 缩放 / 平移　',
        html.B('键盘：'), 'D 删除　U 恢复　Z 撤销',
    ], style={'padding': '4px 16px', 'background': '#fff8e1',
              'fontSize': '11px', 'color': '#5D4037',
              'borderBottom': '1px solid #FFE082'}),

    # ── 6 通道散点图（2列 × 3行）─────────────────────────────────────────
    html.Div(
        [dcc.Graph(
            id=f'graph-ch{ch}',
            config={
                'scrollZoom':             True,
                'displayModeBar':         True,
                'modeBarButtonsToRemove': ['toImage', 'resetScale2d'],
                'displaylogo':            False,
            },
            style={'height': '220px'},
        ) for ch in range(1, 7)],
        style={'display': 'grid',
               'gridTemplateColumns': '1fr 1fr',
               'gap': '4px',
               'padding': '8px'},
    ),

    # ── 数据存储 ──────────────────────────────────────────────────────────
    dcc.Store(id='store-data'),      # dict of lists（紧凑格式）
    dcc.Store(id='store-keep'),      # list[bool]
    dcc.Store(id='store-history'),   # list[list[bool]]（撤销栈）

    # 键盘事件占位
    html.Div(id='kbd-dummy', style={'display': 'none'}),

], style={
    'fontFamily': 'Microsoft YaHei, SimHei, Arial, sans-serif',
    'maxWidth': '1440px',
    'margin': '0 auto',
    'background': 'white',
    'minHeight': '100vh',
})


# ═══════════════════════════════════════════════════════════════════════════
# 键盘快捷键（Clientside Callback）
# ═══════════════════════════════════════════════════════════════════════════
app.clientside_callback(
    """
    function(id) {
        if (window._ae_kbd) return window.dash_clientside.no_update;
        window._ae_kbd = true;
        document.addEventListener('keydown', function(e) {
            if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
            var map = {d: 'btn-delete', u: 'btn-unmark', z: 'btn-undo'};
            var btnId = map[e.key.toLowerCase()];
            if (btnId) { var b = document.getElementById(btnId); if (b) b.click(); }
        });
        return 'ok';
    }
    """,
    Output('kbd-dummy', 'children'),
    Input('kbd-dummy',  'id'),
)


# ═══════════════════════════════════════════════════════════════════════════
# 回调 1：上传文件 → 初始化所有状态
# ═══════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('store-data',    'data'),
    Output('store-keep',    'data'),
    Output('store-history', 'data'),
    Output('file-label',    'children'),
    Output('status-bar',    'children', allow_duplicate=True),
    *[Output(f'graph-ch{ch}', 'figure', allow_duplicate=True) for ch in range(1, 7)],
    Input('upload-file', 'contents'),
    State('upload-file', 'filename'),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if not contents:
        raise PreventUpdate

    # 解码 base64
    _, b64 = contents.split(',', 1)
    text = base64.b64decode(b64).decode('utf-8', errors='replace')

    df = parse_ae_hits(text)
    if len(df) == 0:
        msg = f'❌ 解析失败：{filename} 中未找到有效 hits（请确认格式正确）'
        empty = go.Figure()
        return no_update, no_update, no_update, msg, msg, *([empty] * 6)

    keep    = [True] * len(df)
    history = []
    data_store = {col: df[col].tolist() for col in df.columns}

    figs   = build_figures(data_store, keep)
    n      = len(df)
    ch_cnt = df['CH'].value_counts().sort_index()
    ch_str = '　'.join(f'CH{ch}:{cnt}' for ch, cnt in ch_cnt.items())
    status = f'✅ 已加载：{filename}　共 {n} hits　({ch_str})'
    label  = f'{filename}  ({n} hits)'

    return data_store, keep, history, label, status, *figs


# ═══════════════════════════════════════════════════════════════════════════
# 回调 2：操作按钮 → 更新 keep / history / 图形 / 状态栏
# ═══════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('store-keep',    'data',     allow_duplicate=True),
    Output('store-history', 'data',     allow_duplicate=True),
    Output('status-bar',    'children', allow_duplicate=True),
    *[Output(f'graph-ch{ch}', 'figure', allow_duplicate=True) for ch in range(1, 7)],
    Input('btn-delete', 'n_clicks'),
    Input('btn-unmark',  'n_clicks'),
    Input('btn-undo',    'n_clicks'),
    Input('btn-reset',   'n_clicks'),
    State('store-data',    'data'),
    State('store-keep',    'data'),
    State('store-history', 'data'),
    *[State(f'graph-ch{ch}', 'selectedData') for ch in range(1, 7)],
    prevent_initial_call=True,
)
def on_action(n_del, n_unm, n_undo, n_rst,
              data_store, keep, history,
              *selected_datas):
    if data_store is None or keep is None:
        raise PreventUpdate

    triggered  = ctx.triggered_id
    keep_arr   = np.array(keep, dtype=bool)
    history    = list(history or [])

    if triggered == 'btn-undo':
        if not history:
            raise PreventUpdate
        keep_arr = np.array(history[-1], dtype=bool)
        history  = history[:-1]

    elif triggered == 'btn-reset':
        history  = history[-(MAX_UNDO-1):] + [keep_arr.tolist()]
        keep_arr = np.ones(len(keep_arr), dtype=bool)

    elif triggered in ('btn-delete', 'btn-unmark'):
        # 收集所有通道图中当前选中点的全局索引
        sel_idx = set()
        for sd in selected_datas:
            if sd and 'points' in sd:
                for pt in sd['points']:
                    if 'customdata' in pt:
                        sel_idx.add(int(pt['customdata']))

        if not sel_idx:
            raise PreventUpdate

        # 压栈（限制深度）
        history = history[-(MAX_UNDO-1):] + [keep_arr.tolist()]

        idx_arr = np.array(sorted(sel_idx))
        if triggered == 'btn-delete':
            keep_arr[idx_arr] = False
        else:
            keep_arr[idx_arr] = True

    else:
        raise PreventUpdate

    figs  = build_figures(data_store, keep_arr.tolist())
    n_k   = int(keep_arr.sum())
    n_d   = int((~keep_arr).sum())
    n_all = len(keep_arr)
    pct   = 100. * n_d / n_all if n_all > 0 else 0.
    status = (f'保留 {n_k} hits　|　已删 {n_d} ({pct:.1f}%)　'
              f'|　可撤销步数: {len(history)}')

    return keep_arr.tolist(), history, status, *figs


# ═══════════════════════════════════════════════════════════════════════════
# 回调 3：导出 CSV
# ═══════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('download-csv', 'data'),
    Input('btn-export', 'n_clicks'),
    State('store-data', 'data'),
    State('store-keep', 'data'),
    prevent_initial_call=True,
)
def on_export(n_clicks, data_store, keep):
    if not data_store or keep is None:
        raise PreventUpdate
    df       = pd.DataFrame(data_store)
    keep_arr = np.array(keep, dtype=bool)
    df_clean = df[keep_arr].reset_index(drop=True)
    return dcc.send_data_frame(df_clean.to_csv, 'AE_clean.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    print("=" * 52)
    print("  AE 手动清理工具 — 网页版")
    print("=" * 52)
    print(f"  浏览器访问: http://127.0.0.1:{port}")
    print("  按 Ctrl+C 停止\n")
    app.run(debug=False, host='0.0.0.0', port=port)
