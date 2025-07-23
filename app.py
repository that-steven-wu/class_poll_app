import os
import pandas as pd
import numpy as np

# 强制使用 Agg 后端，避免 Tkinter 错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from flask import Flask, render_template, request, url_for

app = Flask(__name__)
plt.style.use('ggplot')

# 数据存放路径
DATA_DIR           = os.path.join(os.path.dirname(__file__), 'data')
NUMERIC_CSV_PATH   = os.path.join(DATA_DIR, 'submissions.csv')
TEXT_CSV_PATH      = os.path.join(DATA_DIR, 'text_responses.csv')
INFO_CSV_PATH      = os.path.join(DATA_DIR, 'submission_info.csv')
CHART_PATH         = os.path.join('static', 'summary.png')

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

@app.route('/')
def index():
    return render_template('index.html',
                           question_keys=QUESTION_KEYS,
                           method_labels=METHOD_LABELS)

@app.route('/submit', methods=['POST'])
def submit():
    # —— 处理数值题目 ——
    if os.path.exists(NUMERIC_CSV_PATH):
        df_num = pd.read_csv(NUMERIC_CSV_PATH)
    else:
        df_num = pd.DataFrame(columns=['question','method','answer'])
    rows = []
    for q in QUESTION_KEYS:
        for i, m in enumerate(METHOD_LABELS, 1):
            v = request.form.get(f"{q}_answer_{i}")
            try:
                fv = float(v)
            except:
                fv = None
            rows.append({'question': q, 'method': m, 'answer': fv})
    df_num = pd.concat([df_num, pd.DataFrame(rows)], ignore_index=True)
    df_num.to_csv(NUMERIC_CSV_PATH, index=False)

    # —— 处理文本题 Q4, Q5 ——
    q4 = request.form.get('Q4_answer','').strip() or 'N/A'
    q5 = request.form.get('Q5_answer','').strip() or 'N/A'
    if os.path.exists(TEXT_CSV_PATH):
        df_text = pd.read_csv(TEXT_CSV_PATH)
    else:
        df_text = pd.DataFrame(columns=['Q4_answer','Q5_answer'])
    df_text = pd.concat([df_text, pd.DataFrame([{'Q4_answer': q4, 'Q5_answer': q5}])],
                        ignore_index=True)
    df_text.to_csv(TEXT_CSV_PATH, index=False)

    # —— 存储提交详情 ——
    sec  = request.form.get('section','').strip()
    team = request.form.get('team_number','').strip()
    fn   = request.form.get('first_name','').strip()
    ln   = request.form.get('last_name','').strip()
    if os.path.exists(INFO_CSV_PATH):
        df_info = pd.read_csv(INFO_CSV_PATH)
    else:
        df_info = pd.DataFrame(columns=['section','team_number','first_name','last_name'])
    df_info = pd.concat([df_info, pd.DataFrame([{
        'section': sec,
        'team_number': team,
        'first_name': fn,
        'last_name': ln
    }])], ignore_index=True)
    df_info.to_csv(INFO_CSV_PATH, index=False)

    return render_template('thanks.html')

@app.route('/results')
def results():
    # —— Submission Details 分支 ——
    if request.args.get('info') == 'details':
        if os.path.exists(INFO_CSV_PATH):
            df_info = pd.read_csv(INFO_CSV_PATH)
            df_info = df_info.sort_values(
                by=['section','team_number','first_name','last_name'])
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

    # —— 文本视图 Q4/Q5 ——
    text_q = request.args.get('text')
    if text_q in ('Q4','Q5'):
        if os.path.exists(TEXT_CSV_PATH):
            df_text = pd.read_csv(TEXT_CSV_PATH)
            answers = df_text[f"{text_q}_answer"].fillna("N/A").tolist()
        else:
            answers = []
        return render_template('results.html',
                               show_text=True,
                               text_question=text_q,
                               text_answers=answers,
                               show_info=False,
                               message="No text responses yet.")

    # —— 数值视图 Q1–Q3 ——
    group = request.args.get('group','1')
    if group not in GROUP_MAP:
        group = '1'
    keys = GROUP_MAP[group]
    if not os.path.exists(NUMERIC_CSV_PATH):
        return render_template('results.html',
                               show_text=False,
                               show_info=False,
                               chart_url=None,
                               message="There are no submissions yet.",
                               active_group=group)

    df = pd.read_csv(NUMERIC_CSV_PATH)
    n_q, n_m = len(keys), len(METHOD_LABELS)

    # 增大整体画布尺寸，避免子图因间距缩小
    fig_w = n_m * 6   # 每列 6 单位宽
    fig_h = n_q * 4   # 每行 4 单位高
    fig, axes = plt.subplots(n_q, n_m, figsize=(fig_w, fig_h), squeeze=False)

    # 增加左右上下间距与子图间距
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.10,
                        hspace=0.6, wspace=0.6)

    for i, q in enumerate(keys):
        corr = CORRECT_ANSWERS[q]
        lower, upper = corr * 0.98, corr * 1.02
        bins = np.linspace(lower, upper, 21)

        # 统一 y 轴：计算行最大频数
        row_max = 0
        for m in METHOD_LABELS:
            arr = df[(df.question==q)&(df.method==m)]['answer']\
                     .dropna().loc[lambda s:(s>=lower)&(s<=upper)]
            if not arr.empty:
                cnts, _ = np.histogram(arr, bins=bins)
                row_max = max(row_max, cnts.max())

        for j, m in enumerate(METHOD_LABELS):
            ax = axes[i][j]
            data = df[(df.question==q)&(df.method==m)]['answer']\
                     .dropna().loc[lambda s:(s>=lower)&(s<=upper)]

            if not data.empty:
                mean_val = data.mean()
                sd_val   = data.std()
                ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
                ax.axvline(mean_val, color='blue', linestyle='-',
                           linewidth=1.8, alpha=0.6, label='Mean')
            else:
                mean_val = sd_val = 0
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)

            ax.axvline(corr, color='black', linestyle='--',
                       linewidth=1.8, alpha=0.6, label='Correct Answer')

            ax.set_xlim(lower, upper)
            ax.set_xticks([lower, corr, upper])
            ax.xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, pos: f"{x:,.1f}")
            )
            ax.set_ylim(0, row_max * 1.1)
            ax.legend(fontsize=9)
            ax.set_title(f"{q} - {m}", fontsize=14)
            if j == 0:
                ax.set_ylabel('Count', fontsize=12)
            if i == n_q - 1:
                ax.set_xlabel('Answer', fontsize=12)
            ax.tick_params(axis='x', labelrotation=0, labelsize=10)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            # 底部信息行，移动更远些以增大与 x 轴标题间距
            info = f"Mean: {mean_val:.0f}   SD: {sd_val:.1f}   Correct: {corr:.0f}"
            ax.text(0.5, -0.22, info,
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=10, fontweight='bold')

    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    plt.savefig(os.path.join(app.root_path, CHART_PATH),
                dpi=100, bbox_inches='tight')
    plt.close(fig)

    return render_template('results.html',
                           show_text=False,
                           show_info=False,
                           active_group=group,
                           chart_url=url_for('static', filename='summary.png'),
                           message=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=True)
