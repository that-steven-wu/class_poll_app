import os
import pandas as pd
import numpy as np

# 强制使用 Agg 后端，避免 Tkinter 错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator

from flask import Flask, render_template, request, url_for

app = Flask(__name__)
plt.style.use('ggplot')

# 数据存放路径
DATA_DIR       = os.path.join(os.path.dirname(__file__), 'data')
CSV_PATH       = os.path.join(DATA_DIR, 'submissions.csv')
TEXT_CSV_PATH  = os.path.join(DATA_DIR, 'text_responses.csv')
CHART_PATH     = os.path.join('static', 'summary.png')

# 全部问题键
QUESTION_KEYS  = [
    "Q1a","Q1b","Q1c",
    "Q2a","Q2b1","Q2b2",
    "Q3a1","Q3a2","Q3b1","Q3b2"
]
# 分组映射
GROUP_MAP      = {
    '1': ["Q1a","Q1b","Q1c"],
    '2': ["Q2a","Q2b1","Q2b2"],
    '3': ["Q3a1","Q3a2","Q3b1","Q3b2"]
}
# 方法标签
METHOD_LABELS  = ["Not using AI", "Using some AI", "Using refined AI"]

# 标准答案
CORRECT_ANSWERS = {
    "Q1a": 2207682.7,
    "Q1b": 822208.6,
    "Q1c": 5120.3,
    "Q2a": 146640.1,
    "Q2b1": 3330.1,
    "Q2b2": 65.0,
    "Q3a1": 1533691.5,
    "Q3a2": 2297.9,
    "Q3b1": 1768829.7,
    "Q3b2": 14.6
}

os.makedirs(DATA_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html',
                           question_keys=QUESTION_KEYS,
                           method_labels=METHOD_LABELS)

@app.route('/submit', methods=['POST'])
def submit():
    # ——— 处理数值题目 ———
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=['question', 'method', 'answer'])

    new_rows = []
    for q in QUESTION_KEYS:
        for i, method in enumerate(METHOD_LABELS, start=1):
            val = request.form.get(f"{q}_answer_{i}")
            try:
                fv = float(val)
            except (TypeError, ValueError):
                fv = None
            new_rows.append({'question': q, 'method': method, 'answer': fv})

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    # ——— 处理文本题目 Q4 & Q5 ———
    q4 = request.form.get('Q4_answer', '').strip() or 'N/A'
    q5 = request.form.get('Q5_answer', '').strip() or 'N/A'

    if os.path.exists(TEXT_CSV_PATH):
        df_text = pd.read_csv(TEXT_CSV_PATH)
    else:
        df_text = pd.DataFrame(columns=['Q4_answer', 'Q5_answer'])

    df_text = pd.concat(
        [df_text, pd.DataFrame([{'Q4_answer': q4, 'Q5_answer': q5}])],
        ignore_index=True
    )
    df_text.to_csv(TEXT_CSV_PATH, index=False)

    return render_template('thanks.html')

@app.route('/results')
def results():
    # ——— 文本视图分支 ———
    text_q = request.args.get('text')
    if text_q in ('Q4', 'Q5'):
        if os.path.exists(TEXT_CSV_PATH):
            df_text = pd.read_csv(TEXT_CSV_PATH)
            answers = df_text[f"{text_q}_answer"].fillna("N/A").tolist()
        else:
            answers = []
        return render_template('results.html',
                               show_text=True,
                               text_question=text_q,
                               text_answers=answers,
                               message="No text responses yet.")

    # ——— 数值视图分支 ———
    group = request.args.get('group')
    if group not in GROUP_MAP:
        group = '1'
    keys = GROUP_MAP[group]

    if not os.path.exists(CSV_PATH):
        return render_template('results.html',
                               show_text=False,
                               chart_url=None,
                               message="There are no submissions yet.",
                               active_group=group)

    df = pd.read_csv(CSV_PATH)
    n_q, n_m = len(keys), len(METHOD_LABELS)
    fig, axes = plt.subplots(n_q, n_m,
                             figsize=(n_m*5, n_q*3),
                             squeeze=False)

    # 恢复底部留白
    fig.subplots_adjust(bottom=0.10)

    for i, q in enumerate(keys):
        corr = CORRECT_ANSWERS[q]
        # 计算该行最大频数，统一 y 轴
        row_max = 0
        for method in METHOD_LABELS:
            dat = df[(df.question==q)&(df.method==method)]['answer']\
                     .dropna().loc[lambda s: s!=0]
            if not dat.empty:
                counts, _ = np.histogram(dat, bins=12)
                row_max = max(row_max, counts.max())

        for j, method in enumerate(METHOD_LABELS):
            ax = axes[i][j]
            data = (
                df[(df.question==q)&(df.method==method)]['answer']
                  .dropna().loc[lambda s: s!=0]
            )

            if data.empty:
                mean_val = sd_val = 0
                ax.text(0.5, 0.5, 'No data',
                        ha='center', va='center', fontsize=10)
            else:
                mean_val = data.mean()
                sd_val   = data.std()
                # 缩小柱宽
                ax.hist(data,
                        bins=15,
                        rwidth=0.9,
                        edgecolor='black',
                        alpha=0.7)
                ax.axvline(mean_val,
                           color='blue',
                           linestyle='-',
                           linewidth=1.8,
                           alpha=0.6,
                           label='Mean')

            # 绘制正确答案
            ax.axvline(corr,
                       color='black',
                       linestyle='--',
                       linewidth=1.8,
                       alpha=0.6,
                       label='Correct Answer')

            # 统一 y 轴
            ax.set_ylim(0, row_max * 1.1)

            ax.legend(fontsize=9)
            ax.set_title(f"{q} - {method}", fontsize=14)
            if j == 0:
                ax.set_ylabel('Count', fontsize=12)
            if i == n_q - 1:
                ax.set_xlabel('Answer', fontsize=12)

            ax.xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
            )
            locator = MaxNLocator(nbins=4 if corr > 1e6 else 6, integer=True)
            ax.xaxis.set_major_locator(locator)
            ax.tick_params(axis='x', labelrotation=0, labelsize=10)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            # info 行
            info = f"Mean: {mean_val:.1f}   SD: {sd_val:.1f}   Correct: {corr:.1f}"
            ax.text(0.5,
                    -0.32,
                    info,
                    transform=ax.transAxes,
                    ha='center',
                    va='top',
                    fontsize=10,
                    fontweight='bold')

    plt.tight_layout(pad=2.0, w_pad=0.8)
    plt.savefig(os.path.join(app.root_path, CHART_PATH),
                dpi=100, bbox_inches='tight')
    plt.close(fig)

    return render_template('results.html',
                           show_text=False,
                           active_group=group,
                           chart_url=url_for('static', filename='summary.png'),
                           message=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=True)
