#!/usr/bin/env python3
import os, tarfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd, numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from flask import (Flask, render_template, request, url_for,
                   send_from_directory, abort)
from filelock import FileLock
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
plt.style.use('ggplot')

# ========================= 配置区 =========================
DATA_DIR         = os.environ.get("DATA_DIR", "/var/data")
NUMERIC_CSV_PATH = os.path.join(DATA_DIR, 'submissions.csv')
TEXT_CSV_PATH    = os.path.join(DATA_DIR, 'text_responses.csv')
INFO_CSV_PATH    = os.path.join(DATA_DIR, 'submission_info.csv')
LOCK_PATH        = os.path.join(DATA_DIR, 'io.lock')

CLASS_DAY = os.environ.get("CLASS_DAY", "2025-08-04")
CLASS_TZ  = os.environ.get("CLASS_TZ", "America/New_York")
SHOW_MODE = os.environ.get("SHOW_MODE", "class_only")  # 'class_only' or 'all'

# 模板模式（决定前端表单与作图标签）
TEMPLATE_MODE = os.environ.get("TEMPLATE_MODE", "3section")   # '3section' or '2section'

# 去极端值阈值（保留以备后续扩展）
OUTLIER_TRIM_LO_PCT = float(os.environ.get("OUTLIER_TRIM_LO_PCT", "10"))
OUTLIER_TRIM_HI_PCT = float(os.environ.get("OUTLIER_TRIM_HI_PCT", "90"))
TRIM_MIN_TAIL_PCT   = float(os.environ.get("TRIM_MIN_TAIL_PCT", str(OUTLIER_TRIM_LO_PCT)))

# 方案C：离群阈值（相对正确答案的百分比）与统一柱宽
OUTLIER_TOL_PCT = float(os.environ.get("OUTLIER_TOL_PCT", "3.0")) / 100.0  # 3%
BAR_WIDTH       = float(os.environ.get("BAR_WIDTH", "0.3"))                 # 60%

CHART_PATH = os.path.join('static', 'summary.png')

QUESTION_KEYS = [
    "Q1a","Q1b","Q1c",
    "Q2a","Q2b1","Q2b2",
    "Q3a1","Q3a2","Q3b1","Q3b2"
]
GROUP_MAP = {
    '1': ["Q1a","Q1b","Q1c"],
    '2': ["Q2a","Q2b1","Q2b2"],
    '3': ["Q3a1","Q3a2","Q3b1","Q3b2"]
}

# ===== METHOD_LABELS（用于表单与图表显示）=====
if TEMPLATE_MODE == '2section':
    METHOD_LABELS = ["Odd & Even", "Odd Team", "Even Team"]
else:  # '3section'
    METHOD_LABELS = ["Section 1 & 2", "Section 3 & 4", "Section 5 & 6"]

# ===== 历史标签兼容（用于数据聚合） =====
METHOD_SYNONYMS = {
    0: {"Section 1 & 2", "Odd & Even", "Not using AI"},
    1: {"Section 3 & 4", "Odd Team", "Using some AI"},
    2: {"Section 5 & 6", "Even Team", "Using refined AI"},
}

# 正确答案表
CORRECT_ANSWERS = {
    "Q1a": 2207682.7, "Q1b": 822208.6, "Q1c": 5120.3,
    "Q2a": 146640.1, "Q2b1": 3330.1, "Q2b2": 65.0,
    "Q3a1": 1533691.5, "Q3a2": 2297.9, "Q3b1": 1768829.7, "Q3b2": 14.6
}

os.makedirs(DATA_DIR, exist_ok=True)
# ==========================================================

def filter_for_day(df, day_str=CLASS_DAY, tz_str=CLASS_TZ):
    if 'ts' not in df.columns:
        return df.iloc[0:0]
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'], utc=True, errors='coerce')
    start_local = datetime.fromisoformat(day_str).replace(tzinfo=ZoneInfo(tz_str))
    end_local   = start_local.replace(hour=23, minute=59, second=59)
    start_utc   = start_local.astimezone(ZoneInfo("UTC"))
    end_utc     = end_local.astimezone(ZoneInfo("UTC"))
    return df[(df['ts'] >= start_utc) & (df['ts'] <= end_utc)]

@app.route('/')
def index():
    # 前端根据 disable_first 控制 2section
    return render_template(
        'index.html',
        question_keys=QUESTION_KEYS,
        method_labels=METHOD_LABELS,
        disable_first=(TEMPLATE_MODE == '2section')
    )

@app.route('/submit', methods=['POST'])
def submit():
    now_utc = datetime.now(timezone.utc).isoformat()
    with FileLock(LOCK_PATH, timeout=10):
        # 数值题保存
        df_num = pd.read_csv(NUMERIC_CSV_PATH) if os.path.exists(NUMERIC_CSV_PATH) else pd.DataFrame(columns=['question','method','answer','ts'])

        # 2section：只接收 Odd Team / Even Team（表单索引 2、3）；3section：三列全接收
        if TEMPLATE_MODE == '2section':
            method_positions = [2, 3]
        else:
            method_positions = [1, 2, 3]

        rows = []
        for q in QUESTION_KEYS:
            for pos in method_positions:  # 1-based for form keys
                m = METHOD_LABELS[pos - 1]
                key = f"{q}_answer_{pos}"
                try:
                    fv_raw = request.form.get(key)
                    fv = float(fv_raw) if fv_raw not in (None, '') else None
                except Exception:
                    fv = None
                rows.append({'question': q, 'method': m, 'answer': fv, 'ts': now_utc})
        df_num = pd.concat([df_num, pd.DataFrame(rows)], ignore_index=True)
        df_num.to_csv(NUMERIC_CSV_PATH, index=False)

        # 文本题保存
        q4 = request.form.get('Q4_answer','').strip() or 'N/A'
        q5 = request.form.get('Q5_answer','').strip() or 'N/A'
        df_text = pd.read_csv(TEXT_CSV_PATH) if os.path.exists(TEXT_CSV_PATH) else pd.DataFrame(columns=['Q4_answer','Q5_answer','ts'])
        df_text = pd.concat([df_text, pd.DataFrame([{'Q4_answer':q4,'Q5_answer':q5,'ts':now_utc}])], ignore_index=True)
        df_text.to_csv(TEXT_CSV_PATH, index=False)

        # 基本信息保存
        info = {k: request.form.get(k,'').strip() for k in ['section','team_number','first_name','last_name']}
        df_info = pd.read_csv(INFO_CSV_PATH) if os.path.exists(INFO_CSV_PATH) else pd.DataFrame(columns=['section','team_number','first_name','last_name','ts'])
        df_info = pd.concat([df_info, pd.DataFrame([{**info,'ts':now_utc}])], ignore_index=True)
        df_info.to_csv(INFO_CSV_PATH, index=False)

    return render_template('thanks.html')

def _get_series_by_method(df: pd.DataFrame, qkey: str, method_col_idx: int) -> pd.Series:
    """根据同义集合抓取某题-某方式的答案（去掉NaN）。"""
    accepted = METHOD_SYNONYMS[method_col_idx]
    s = df[(df.question == qkey) & (df.method.isin(accepted))]['answer']
    return s.dropna()

def _nonzero_series(s: pd.Series) -> pd.Series:
    """统一在统计时排除 0（视为无效）。"""
    return s[(s.notna()) & (s != 0)]

def _pct_formatter():
    return mtick.PercentFormatter(xmax=1.0, decimals=0)

@app.route('/results')
def results():
    # 学生信息详情
    if request.args.get('info') == 'details':
        if os.path.exists(INFO_CSV_PATH):
            with FileLock(LOCK_PATH, timeout=10):
                df_info = pd.read_csv(INFO_CSV_PATH)
            if SHOW_MODE == 'class_only':
                df_info = filter_for_day(df_info)
            df_info = df_info.sort_values(by=['section','team_number','first_name','last_name'])
            info_rows = df_info.to_dict(orient='records')
        else:
            info_rows = []
        return render_template('results.html',
                               show_info=True,
                               info_rows=info_rows,
                               show_text=False,
                               chart_url=None,
                               active_group=None,
                               message=None)

    # 文本题展示
    text_q = request.args.get('text')
    if text_q in ('Q4', 'Q5'):
        if os.path.exists(TEXT_CSV_PATH):
            with FileLock(LOCK_PATH, timeout=10):
                df_text = pd.read_csv(TEXT_CSV_PATH)
            if SHOW_MODE == 'class_only':
                df_text = filter_for_day(df_text)
            answers = df_text[f"{text_q}_answer"].fillna('N/A').tolist()
        else:
            answers = []
        return render_template('results.html',
                               show_text=True,
                               text_question=text_q,
                               text_answers=answers,
                               show_info=False,
                               chart_url=None,
                               active_group=None,
                               message=None)

    # 数值题结果
    group = request.args.get('group', '1')
    if group not in GROUP_MAP:
        group = '1'
    keys = GROUP_MAP[group]

    if not os.path.exists(NUMERIC_CSV_PATH):
        return render_template('results.html', show_text=False, show_info=False,
                               chart_url=None, message="No submissions yet.", active_group=group)
    with FileLock(LOCK_PATH, timeout=10):
        df = pd.read_csv(NUMERIC_CSV_PATH)
    if SHOW_MODE == 'class_only':
        df = filter_for_day(df)
    if df.empty:
        return render_template('results.html', show_text=False, show_info=False,
                               chart_url=None, message="No data for selected day.", active_group=group)

    # ============== 新版绘图：每题三面板 ==============
    n_q = len(keys)
    n_panels = 3  # 1:正确率  2:均值vs正确  3:离群比例
    fig_w, fig_h = 6*n_panels, 3.8*n_q
    fig, axes = plt.subplots(n_q, n_panels, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.12)

    # 决定要绘制的列与标签
    if TEMPLATE_MODE == '2section':
        method_idx_for_plot = [1, 2]  # 只画 Odd Team / Even Team
    else:
        method_idx_for_plot = [0, 1, 2]
    plot_labels = [METHOD_LABELS[k] for k in method_idx_for_plot]
    x_idx = np.arange(len(method_idx_for_plot))

    for i, q in enumerate(keys):
        corr = CORRECT_ANSWERS.get(q, np.nan)

        # 取各方式原始序列（按所选列）
        series_by_method_all = [_get_series_by_method(df, q, k) for k in method_idx_for_plot]
        series_by_method     = [_nonzero_series(s) for s in series_by_method_all]

        # ========== 面板1：正确率（±0.5%），0 已排除 ==========
        ax1 = axes[i][0]
        correct_pct, n_valid = [], []
        for s in series_by_method:
            s_valid = s
            n_valid.append(len(s_valid))
            if np.isnan(corr) or len(s_valid) == 0:
                correct_pct.append(0.0)
            else:
                tol = 0.005 * abs(corr)
                m = ((s_valid >= corr - tol) & (s_valid <= corr + tol)).sum()
                correct_pct.append(m / len(s_valid))

        bars1 = ax1.bar(x_idx, correct_pct, width=BAR_WIDTH, edgecolor='black', alpha=0.8)
        ax1.set_ylim(0, 1)
        ax1.yaxis.set_major_formatter(_pct_formatter())
        ax1.set_xticks(x_idx, plot_labels)
        ax1.set_title(f"{q} — Accuracy (±0.5%)", fontsize=12)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
        for rect, p in zip(bars1, correct_pct):
            ax1.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.02,
                     f"{p:.0%}", ha='center', va='bottom', fontsize=9)

        # ========== 面板2：均值（3%–97% 截尾） vs 正确值，0 已排除 ==========
        ax2 = axes[i][1]
        means_trim = []
        for s in series_by_method:
            if len(s) == 0:
                means_trim.append(np.nan)
            else:
                lo, hi = np.percentile(s, [3, 97])
                s_trim = s[(s >= lo) & (s <= hi)]
                means_trim.append(s_trim.mean() if len(s_trim) > 0 else np.nan)

        heights = [corr] + means_trim


        labels  = ["Correct"] + plot_labels
        x2 = np.arange(len(heights))

        bars2 = ax2.bar(x2, heights, width=BAR_WIDTH, edgecolor='black', alpha=0.85)

        if np.isfinite(corr):
            ax2.axhline(corr, color='blue', linestyle='--', linewidth=1.6, alpha=0.9)

        # 先设置普通标签
        ax2.set_xticks(x2, labels, rotation=0)

        # 再单独修改第一个标签 (Correct) 的样式
        xtick_texts = ax2.get_xticklabels()
        if xtick_texts:
            xtick_texts[0].set_color('blue')
            xtick_texts[0].set_fontweight('bold')



        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, pos: f"{v:,.1f}"))
        ax2.set_title(f"{q} — Mean (trim 3–97%) vs Correct", fontsize=12)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

        finite_vals = [v for v in heights if np.isfinite(v)]
        if len(finite_vals) >= 2:
            vmin, vmax = min(finite_vals), max(finite_vals)
            if vmax > vmin:
                pad = 0.10 * (vmax - vmin)
                ax2.set_ylim(vmin - pad, vmax + pad)

        for rect, val in zip(bars2, heights):
            if np.isfinite(val):
                ax2.text(rect.get_x()+rect.get_width()/2, rect.get_height(),
                         f"{val:,.1f}", ha='center', va='bottom', fontsize=9)

        # ========== 面板3：离群比例（> 正确值的 3%），0 已排除 ==========
        ax3 = axes[i][2]
        if np.isnan(corr) or corr == 0:
            outlier_pct = [0.0] * len(series_by_method)
        else:
            thr = OUTLIER_TOL_PCT * abs(corr)
            outlier_pct = [
                (np.abs(s - corr) > thr).sum() / len(s) if len(s) > 0 else 0.0
                for s in series_by_method
            ]

        bars3 = ax3.bar(x_idx, outlier_pct, width=BAR_WIDTH, edgecolor='black', alpha=0.8)
        ax3.set_ylim(0, 1)
        ax3.yaxis.set_major_formatter(_pct_formatter())
        ax3.set_xticks(x_idx, plot_labels)
        ax3.set_title(f"{q} — Outlier Rate (> {int(OUTLIER_TOL_PCT*100)}% from Correct)", fontsize=12)
        ax3.grid(True, axis='y', linestyle='--', alpha=0.6)
        for rect, p in zip(bars3, outlier_pct):
            ax3.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.02,
                     f"{p:.0%}", ha='center', va='bottom', fontsize=9)

        # 行尾样本量（非零有效 N）
        ns_text = "Ns (nonzero): " + ", ".join(
            f"{lbl}={len(series_by_method[i])}" for i, lbl in enumerate(plot_labels)
        )
        axes[i][0].text(0.02, -0.28, ns_text,
                        transform=axes[i][0].transAxes, fontsize=9, ha='left', va='top')

        # 最后一行加上坐标轴标签
        if i == n_q - 1:
            axes[i][0].set_xlabel('Method')
            axes[i][1].set_xlabel('Value')
            axes[i][2].set_xlabel('Method')

        # 第一列加上 y 轴标签
        axes[i][0].set_ylabel('Percent')

    with FileLock(LOCK_PATH, timeout=10):
        fig.savefig(os.path.join(app.root_path, CHART_PATH), dpi=180)
    plt.close(fig)
    # ============== 绘图结束 ==============

    return render_template('results.html', show_text=False, show_info=False,
                           chart_url=url_for('static', filename='summary.png'), active_group=group, message=None)

@app.route('/download')
def download_data():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    fname = f"data_{ts}.tgz"
    out_path = os.path.join('static', fname)
    with FileLock(LOCK_PATH, timeout=10):
        with tarfile.open(out_path, 'w:gz') as tar:
            tar.add(DATA_DIR, arcname=os.path.basename(DATA_DIR))
    if os.path.exists(out_path):
        return send_from_directory('static', fname, as_attachment=True)
    abort(404)

if __name__ == '__main__':
    # 启动时打印当前模式与阈值，便于确认
    print("TEMPLATE_MODE =", TEMPLATE_MODE)
    print("OUTLIER_TRIM_LO_PCT =", OUTLIER_TRIM_LO_PCT)
    print("OUTLIER_TRIM_HI_PCT =", OUTLIER_TRIM_HI_PCT)
    print("TRIM_MIN_TAIL_PCT   =", TRIM_MIN_TAIL_PCT)
    print("OUTLIER_TOL_PCT     =", OUTLIER_TOL_PCT)
    print("BAR_WIDTH           =", BAR_WIDTH)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
