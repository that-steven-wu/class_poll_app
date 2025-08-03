# #!/usr/bin/env python3
# import os
# import tarfile
# from datetime import datetime, timezone
# from zoneinfo import ZoneInfo

# import pandas as pd
# import numpy as np

# # 强制使用 Agg 后端，避免 Tkinter 错误
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick

# from flask import Flask, render_template, request, url_for, send_from_directory, abort
# from filelock import FileLock
# from dotenv import load_dotenv
# load_dotenv()

# app = Flask(__name__)
# plt.style.use('ggplot')

# # ========================= 配置区 =========================
# DATA_DIR         = os.environ.get("DATA_DIR", "/var/data")
# NUMERIC_CSV_PATH = os.path.join(DATA_DIR, 'submissions.csv')
# TEXT_CSV_PATH    = os.path.join(DATA_DIR, 'text_responses.csv')
# INFO_CSV_PATH    = os.path.join(DATA_DIR, 'submission_info.csv')
# LOCK_PATH        = os.path.join(DATA_DIR, 'io.lock')

# CLASS_DAY = os.environ.get("CLASS_DAY", "2025-08-04")
# CLASS_TZ  = os.environ.get("CLASS_TZ", "America/New_York")
# SHOW_MODE = os.environ.get("SHOW_MODE", "class_only")  # 'class_only' or 'all'

# CHART_PATH = os.path.join('static', 'summary.png')

# QUESTION_KEYS = [
#     "Q1a","Q1b","Q1c",
#     "Q2a","Q2b1","Q2b2",
#     "Q3a1","Q3a2","Q3b1","Q3b2"
# ]
# GROUP_MAP = {
#     '1': ["Q1a","Q1b","Q1c"],
#     '2': ["Q2a","Q2b1","Q2b2"],
#     '3': ["Q3a1","Q3a2","Q3b1","Q3b2"]
# }
# METHOD_LABELS = ["Not using AI", "Using some AI", "Using refined AI"]
# CORRECT_ANSWERS = {
#     "Q1a": 2207682.7, "Q1b": 822208.6, "Q1c": 5120.3,
#     "Q2a": 146640.1, "Q2b1": 3330.1, "Q2b2": 65.0,
#     "Q3a1": 1533691.5, "Q3a2": 2297.9, "Q3b1": 1768829.7, "Q3b2": 14.6
# }

# os.makedirs(DATA_DIR, exist_ok=True)
# # ==========================================================

# def filter_for_day(df, day_str=CLASS_DAY, tz_str=CLASS_TZ):
#     """只保留课堂当天（本地时区 day_str 的 00:00~23:59:59）"""
#     if 'ts' not in df.columns:
#         return df.iloc[0:0]
#     df = df.copy()
#     df['ts'] = pd.to_datetime(df['ts'], utc=True, errors='coerce')
#     start_local = datetime.fromisoformat(day_str).replace(tzinfo=ZoneInfo(tz_str))
#     end_local   = start_local.replace(hour=23, minute=59, second=59)
#     start_utc   = start_local.astimezone(ZoneInfo("UTC"))
#     end_utc     = end_local.astimezone(ZoneInfo("UTC"))
#     return df[(df['ts'] >= start_utc) & (df['ts'] <= end_utc)]

# @app.route('/')
# def index():
#     return render_template('index.html',
#                            question_keys=QUESTION_KEYS,
#                            method_labels=METHOD_LABELS)

# @app.route('/submit', methods=['POST'])
# def submit():
#     now_utc = datetime.now(timezone.utc).isoformat()
#     with FileLock(LOCK_PATH, timeout=10):
#         # 数值题保存
#         df_num = pd.read_csv(NUMERIC_CSV_PATH) if os.path.exists(NUMERIC_CSV_PATH) else pd.DataFrame(columns=['question','method','answer','ts'])
#         rows = []
#         for q in QUESTION_KEYS:
#             for i, m in enumerate(METHOD_LABELS, 1):
#                 v = request.form.get(f"{q}_answer_{i}")
#                 try:
#                     fv = float(v)
#                 except:
#                     fv = None
#                 rows.append({'question':q,'method':m,'answer':fv,'ts':now_utc})
#         df_num = pd.concat([df_num, pd.DataFrame(rows)], ignore_index=True)
#         df_num.to_csv(NUMERIC_CSV_PATH, index=False)

#         # 文本题保存
#         q4 = request.form.get('Q4_answer','').strip() or 'N/A'
#         q5 = request.form.get('Q5_answer','').strip() or 'N/A'
#         df_text = pd.read_csv(TEXT_CSV_PATH) if os.path.exists(TEXT_CSV_PATH) else pd.DataFrame(columns=['Q4_answer','Q5_answer','ts'])
#         df_text = pd.concat([df_text, pd.DataFrame([{'Q4_answer':q4,'Q5_answer':q5,'ts':now_utc}])], ignore_index=True)
#         df_text.to_csv(TEXT_CSV_PATH, index=False)

#         # 基本信息保存
#         info = {k: request.form.get(k,'').strip() for k in ['section','team_number','first_name','last_name']}
#         df_info = pd.read_csv(INFO_CSV_PATH) if os.path.exists(INFO_CSV_PATH) else pd.DataFrame(columns=['section','team_number','first_name','last_name','ts'])
#         df_info = pd.concat([df_info, pd.DataFrame([{**info,'ts':now_utc}])], ignore_index=True)
#         df_info.to_csv(INFO_CSV_PATH, index=False)

#     return render_template('thanks.html')

# @app.route('/results')
# def results():
#     # Submission Details
#     if request.args.get('info') == 'details':
#         if os.path.exists(INFO_CSV_PATH):
#             with FileLock(LOCK_PATH, timeout=10):
#                 df_info = pd.read_csv(INFO_CSV_PATH)
#             if SHOW_MODE == "class_only":
#                 df_info = filter_for_day(df_info)
#             df_info = df_info.sort_values(by=['section','team_number','first_name','last_name'])
#             info_rows = df_info.to_dict(orient='records')
#         else:
#             info_rows = []
#         return render_template('results.html',
#                                show_info=True,
#                                info_rows=info_rows,
#                                show_text=False,
#                                chart_url=None,
#                                active_group=None,
#                                message=None)

#     # 文本题
#     text_q = request.args.get('text')
#     if text_q in ('Q4', 'Q5'):
#         if os.path.exists(TEXT_CSV_PATH):
#             with FileLock(LOCK_PATH, timeout=10):
#                 df_text = pd.read_csv(TEXT_CSV_PATH)
#             if SHOW_MODE == "class_only":
#                 df_text = filter_for_day(df_text)
#             answers = df_text[f"{text_q}_answer"].fillna("N/A").tolist()
#         else:
#             answers = []
#         return render_template('results.html',
#                                show_text=True,
#                                text_question=text_q,
#                                text_answers=answers,
#                                show_info=False,
#                                message="No text responses yet.")

#     # 数值题
#     group = request.args.get('group', '1')
#     if group not in GROUP_MAP:
#         group = '1'
#     keys = GROUP_MAP[group]

#     if not os.path.exists(NUMERIC_CSV_PATH):
#         return render_template('results.html',
#                                show_text=False,
#                                show_info=False,
#                                chart_url=None,
#                                message="There are no submissions yet.",
#                                active_group=group)

#     with FileLock(LOCK_PATH, timeout=10):
#         df = pd.read_csv(NUMERIC_CSV_PATH)
#     if SHOW_MODE == "class_only":
#         df = filter_for_day(df)
#     if df.empty:
#         return render_template('results.html',
#                                show_text=False,
#                                show_info=False,
#                                chart_url=None,
#                                message="No data for selected day.",
#                                active_group=group)

#     n_q, n_m = len(keys), len(METHOD_LABELS)

#     # ========= 画图 =========
#     fig_w = n_m * 6
#     fig_h = n_q * 4
#     fig, axes = plt.subplots(n_q, n_m,
#                              figsize=(fig_w, fig_h),
#                              squeeze=False,
#                              constrained_layout=True)
#     fig.set_constrained_layout_pads(hspace=0.1)

#     for i, q in enumerate(keys):
#         corr = CORRECT_ANSWERS[q]
#         lower, upper = corr * 0.97, corr * 1.03
#         bins = np.linspace(lower, upper, 21)

#         # 统一 y 轴高度
#         row_max = 0
#         for m in METHOD_LABELS:
#             arr = df[(df.question == q) & (df.method == m)]['answer'] \
#                     .dropna().loc[lambda s: (s >= lower) & (s <= upper)]
#             if not arr.empty:
#                 cnts, _ = np.histogram(arr, bins=bins)
#                 row_max = max(row_max, cnts.max())
#         if row_max == 0:
#             row_max = 1

#         for j, m in enumerate(METHOD_LABELS):
#             ax = axes[i][j]
#             # 原 data 保持：只画正确答案附近的直方图
#             data = df[(df.question == q) & (df.method == m)]['answer'] \
#                     .dropna().loc[lambda s: (s >= lower) & (s <= upper)]

#             # 新：先剔除 0 再计算均值和标准差
#             all_vals = df[(df.question == q) & (df.method == m)]['answer'].dropna()
#             nonzero  = all_vals[all_vals != 0]

#             if not nonzero.empty:
#                 mean_val = nonzero.mean()
#                 sd_val   = nonzero.std()
#             else:
#                 mean_val = sd_val = 0

#             if not data.empty:
#                 ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
#                 ax.axvline(mean_val, color='blue', linestyle='-',
#                            linewidth=1.8, alpha=0.6, label='Mean')
#             else:
#                 ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)

#             ax.axvline(corr, color='black', linestyle='--',
#                        linewidth=1.8, alpha=0.6, label='Correct Answer')

#             ax.set_xlim(lower, upper)
#             ax.set_xticks([lower, corr, upper])
#             ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x:,.1f}"))
#             ax.set_ylim(0, row_max * 1.1)
#             ax.legend(fontsize=9)
#             ax.set_title(f"{q} - {m}", fontsize=14)
#             if j == 0:
#                 ax.set_ylabel('Count', fontsize=12)
#             if i == n_q - 1:
#                 ax.set_xlabel('Answer', fontsize=12)
#             ax.tick_params(axis='x', labelrotation=0, labelsize=10)
#             ax.grid(True, axis='y', linestyle='--', alpha=0.6)

#             info = f"Mean: {mean_val:.1f}   SD: {sd_val:.1f}   Correct: {corr:.1f}"
#             ax.text(0.5, -0.2, info,
#                     transform=ax.transAxes, ha='center', va='top',
#                     fontsize=10, fontweight='bold')

#     # 保存图表
#     with FileLock(LOCK_PATH, timeout=10):
#         fig.savefig(os.path.join(app.root_path, CHART_PATH), dpi=180)
#     plt.close(fig)

#     return render_template('results.html',
#                            show_text=False,
#                            show_info=False,
#                            active_group=group,
#                            chart_url=url_for('static', filename='summary.png'),
#                            message=None)

# @app.route('/download')
# def download_data():
#     """
#     动态打包 DATA_DIR 下所有内容，生成带时间戳的 tgz 并提供下载
#     """
#     ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
#     fname = f"data_{ts}.tgz"
#     out_path = os.path.join('static', fname)
#     # 加锁保证数据一致
#     with FileLock(LOCK_PATH, timeout=10):
#         with tarfile.open(out_path, 'w:gz') as tar:
#             tar.add(DATA_DIR, arcname=os.path.basename(DATA_DIR))
#     if os.path.exists(out_path):
#         return send_from_directory('static', fname, as_attachment=True)
#     else:
#         abort(404)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)), debug=True)





# Version 2

#!/usr/bin/env python3
import os
import tarfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

# 强制使用 Agg 后端，避免 Tkinter 错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from flask import Flask, render_template, request, url_for, send_from_directory, abort
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
METHOD_LABELS = ["Not using AI", "Using some AI", "Using refined AI"]
CORRECT_ANSWERS = {
    "Q1a": 2207682.7, "Q1b": 822208.6, "Q1c": 5120.3,
    "Q2a": 146640.1, "Q2b1": 3330.1, "Q2b2": 65.0,
    "Q3a1": 1533691.5, "Q3a2": 2297.9, "Q3b1": 1768829.7, "Q3b2": 14.6
}

os.makedirs(DATA_DIR, exist_ok=True)
# ==========================================================

def filter_for_day(df, day_str=CLASS_DAY, tz_str=CLASS_TZ):
    """只保留课堂当天（本地时区 day_str 的 00:00~23:59:59）"""
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
    return render_template('index.html',
                           question_keys=QUESTION_KEYS,
                           method_labels=METHOD_LABELS)

@app.route('/submit', methods=['POST'])
def submit():
    now_utc = datetime.now(timezone.utc).isoformat()
    with FileLock(LOCK_PATH, timeout=10):
        df_num = pd.read_csv(NUMERIC_CSV_PATH) if os.path.exists(NUMERIC_CSV_PATH) else pd.DataFrame(columns=['question','method','answer','ts'])
        rows = []
        for q in QUESTION_KEYS:
            for i, m in enumerate(METHOD_LABELS, 1):
                v = request.form.get(f"{q}_answer_{i}")
                try:
                    fv = float(v)
                except:
                    fv = None
                rows.append({'question': q, 'method': m, 'answer': fv, 'ts': now_utc})
        df_num = pd.concat([df_num, pd.DataFrame(rows)], ignore_index=True)
        df_num.to_csv(NUMERIC_CSV_PATH, index=False)

        q4 = request.form.get('Q4_answer','').strip() or 'N/A'
        q5 = request.form.get('Q5_answer','').strip() or 'N/A'
        df_text = pd.read_csv(TEXT_CSV_PATH) if os.path.exists(TEXT_CSV_PATH) else pd.DataFrame(columns=['Q4_answer','Q5_answer','ts'])
        df_text = pd.concat([df_text, pd.DataFrame([{'Q4_answer': q4, 'Q5_answer': q5, 'ts': now_utc}])], ignore_index=True)
        df_text.to_csv(TEXT_CSV_PATH, index=False)

        info = {k: request.form.get(k,'').strip() for k in ['section','team_number','first_name','last_name']}
        df_info = pd.read_csv(INFO_CSV_PATH) if os.path.exists(INFO_CSV_PATH) else pd.DataFrame(columns=['section','team_number','first_name','last_name','ts'])
        df_info = pd.concat([df_info, pd.DataFrame([{**info, 'ts': now_utc}])], ignore_index=True)
        df_info.to_csv(INFO_CSV_PATH, index=False)

    return render_template('thanks.html')

@app.route('/results')
def results():
    if request.args.get('info') == 'details':
        if os.path.exists(INFO_CSV_PATH):
            with FileLock(LOCK_PATH, timeout=10):
                df_info = pd.read_csv(INFO_CSV_PATH)
            if SHOW_MODE == "class_only":
                df_info = filter_for_day(df_info)
            df_info = df_info.sort_values(by=['section','team_number','first_name','last_name'])
            info_rows = df_info.to_dict(orient='records')
        else:
            info_rows = []
        return render_template('results.html', show_info=True, info_rows=info_rows, show_text=False, chart_url=None, active_group=None, message=None)

    text_q = request.args.get('text')
    if text_q in ('Q4', 'Q5'):
        if os.path.exists(TEXT_CSV_PATH):
            with FileLock(LOCK_PATH, timeout=10):
                df_text = pd.read_csv(TEXT_CSV_PATH)
            if SHOW_MODE == "class_only":
                df_text = filter_for_day(df_text)
            answers = df_text[f"{text_q}_answer"].fillna("N/A").tolist()
        else:
            answers = []
        return render_template('results.html', show_text=True, text_question=text_q, text_answers=answers, show_info=False, message="No text responses yet.")

    group = request.args.get('group', '1')
    if group not in GROUP_MAP:
        group = '1'
    keys = GROUP_MAP[group]

    if not os.path.exists(NUMERIC_CSV_PATH):
        return render_template('results.html', show_text=False, show_info=False, chart_url=None, message="There are no submissions yet.", active_group=group)

    with FileLock(LOCK_PATH, timeout=10):
        df = pd.read_csv(NUMERIC_CSV_PATH)
    if SHOW_MODE == "class_only":
        df = filter_for_day(df)
    if df.empty:
        return render_template('results.html', show_text=False, show_info=False, chart_url=None, message="No data for selected day.", active_group=group)

    n_q, n_m = len(keys), len(METHOD_LABELS)

    # ========= 画图 =========
    fig_w = n_m * 6
    fig_h = n_q * 4
    fig, axes = plt.subplots(n_q, n_m, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.1)

    for i, q in enumerate(keys):
        corr = CORRECT_ANSWERS[q]
        lower, upper = corr * 0.98, corr * 1.02  # 固定 ±2% 区间
        bins = np.linspace(lower, upper, 21)

        # 统一 y 轴高度
        row_max = 0
        for m in METHOD_LABELS:
            vals = df[(df.question == q) & (df.method == m)]['answer'].dropna()
            nonzero = vals[vals != 0]
            data_vals = nonzero[(nonzero >= lower) & (nonzero <= upper)]
            if not data_vals.empty:
                cnts, _ = np.histogram(data_vals, bins=bins)
                row_max = max(row_max, cnts.max())
        if row_max == 0:
            row_max = 1

        for j, m in enumerate(METHOD_LABELS):
            ax = axes[i][j]
            vals = df[(df.question == q) & (df.method == m)]['answer'].dropna()
            nonzero = vals[vals != 0]

            # 全体非零值 mean/SD
            if not nonzero.empty:
                mean_val = nonzero.mean()
                sd_val   = nonzero.std()
            else:
                mean_val = sd_val = 0

            data = nonzero[(nonzero >= lower) & (nonzero <= upper)]
            if not data.empty:
                ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)

            # Correct Answer
            ax.axvline(corr, color='black', linestyle='--', linewidth=1.8, alpha=0.6, label='Correct Answer')

            # Mean: 若在区间内直接画线，超出则画在边界并标注
            if lower <= mean_val <= upper:
                ax.axvline(mean_val, color='blue', linestyle='-', linewidth=1.8, alpha=0.6, label='Mean')
            else:
                boundary = lower if mean_val < lower else upper
                ax.axvline(boundary, color='blue', linestyle='-', linewidth=1.8, alpha=0.6)
                # 文本箭头标注
                label = f'mean={mean_val:.1f}'
                if mean_val < lower:
                    ax.text(boundary, row_max * 1.05, f'← {label}', ha='left', va='bottom', fontsize=9)
                else:
                    ax.text(boundary, row_max * 1.05, f'{label} →', ha='right', va='bottom', fontsize=9)

            ax.set_xlim(lower, upper)
            ax.set_xticks([lower, corr, upper])
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x:,.1f}"))
            ax.set_ylim(0, row_max * 1.1)
            ax.legend(fontsize=9)
            ax.set_title(f"{q} - {m}", fontsize=14)
            if j == 0:
                ax.set_ylabel('Count', fontsize=12)
            if i == n_q - 1:
                ax.set_xlabel('Answer', fontsize=12)
            ax.tick_params(axis='x', labelrotation=0, labelsize=10)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            info = f"Mean: {mean_val:.1f}   SD: {sd_val:.1f}   Correct: {corr:.1f}"
            ax.text(0.5, -0.2, info, transform=ax.transAxes, ha='center', va='top', fontsize=10, fontweight='bold')

    # 保存图表
    with FileLock(LOCK_PATH, timeout=10):
        fig.savefig(os.path.join(app.root_path, CHART_PATH), dpi=180)
    plt.close(fig)

    return render_template('results.html', show_text=False, show_info=False, active_group=group, chart_url=url_for('static', filename='summary.png'), message=None)

@app.route('/download')
def download_data():
    """
    动态打包 DATA_DIR 下所有内容，生成带时间戳的 tgz 并提供下载
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    fname = f"data_{ts}.tgz"
    out_path = os.path.join('static', fname)
    with FileLock(LOCK_PATH, timeout=10):
        with tarfile.open(out_path, 'w:gz') as tar:
            tar.add(DATA_DIR, arcname=os.path.basename(DATA_DIR))
    if os.path.exists(out_path):
        return send_from_directory('static', fname, as_attachment=True)
    else:
        abort(404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5000)), debug=True)
