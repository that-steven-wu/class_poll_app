import os
import pandas as pd

# 强制使用 Agg 后端，避免 Tkinter 错误
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator

from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# 使用内置 ggplot 风格增强可读性
plt.style.use('ggplot')

# 数据存放路径
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
CSV_PATH   = os.path.join(DATA_DIR, 'submissions.csv')
CHART_PATH = os.path.join('static', 'summary.png')

# 全部问题键
QUESTION_KEYS = [
    "Q1a","Q1b","Q1c",
    "Q2a","Q2b1","Q2b2",
    "Q3a1","Q3a2","Q3b1","Q3b2"
]
# 分组映射
GROUP_MAP = {
    '1': ["Q1a","Q1b","Q1c"],
    '2': ["Q2a","Q2b1","Q2b2"],
    '3': ["Q3a1","Q3a2","Q3b1","Q3b2"]
}
# 三种方法标签
METHOD_LABELS = ["Not using AI", "Using some AI", "Using refined AI"]

# 标准答案（去掉千分位逗号）
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

# 确保 data 目录存在
os.makedirs(DATA_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', question_keys=QUESTION_KEYS, method_labels=METHOD_LABELS)

@app.route('/submit', methods=['POST'])
def submit():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=['question', 'method', 'answer'])

    new_rows = []
    for q in QUESTION_KEYS:
        for i, method in enumerate(METHOD_LABELS, start=1):
            answer = request.form.get(f"{q}_answer_{i}")
            try:
                answer_val = float(answer)
            except (TypeError, ValueError):
                answer_val = None
            new_rows.append({'question': q, 'method': method, 'answer': answer_val})

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    return render_template('thanks.html')

@app.route('/results')
def results():
    # 默认 Q1
    group = request.args.get('group')
    if group not in GROUP_MAP:
        group = '1'
    keys = GROUP_MAP[group]

    if not os.path.exists(CSV_PATH):
        return render_template('results.html', chart_url=None,
                               message="There are no submissions yet.",
                               active_group=group)

    df = pd.read_csv(CSV_PATH)
    n_questions = len(keys)
    n_methods   = len(METHOD_LABELS)

    fig_w, fig_h = n_methods * 5, n_questions * 3
    fig, axes = plt.subplots(n_questions, n_methods, figsize=(fig_w, fig_h), squeeze=False)

    for i, q in enumerate(keys):
        correct = CORRECT_ANSWERS.get(q)
        for j, method in enumerate(METHOD_LABELS):
            ax = axes[i][j]
            data = df[(df['question']==q)&(df['method']==method)]['answer'].dropna()
            if not data.empty:
                ax.hist(data, bins=12, edgecolor='black', alpha=0.7)
                ax.axvline(data.mean(), color='blue', linestyle='-', linewidth=1.8, alpha=0.6, label='Mean')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=10)
            if correct is not None:
                ax.axvline(correct, color='black', linestyle='--', linewidth=1.8, alpha=0.6, label='Correct Answer')
            if data.notna().any() or correct is not None:
                ax.legend(fontsize=8)

            # 增大标题并设置字体
            ax.set_title(f"{q} - {method}", fontsize=14)
            if j == 0:
                ax.set_ylabel('Count', fontsize=12)
            if i == n_questions-1:
                ax.set_xlabel('Answer', fontsize=12)

            # 格式化 x 轴：千分位 & 水平显示
            ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{int(x):,}"))
            ax.tick_params(axis='x', labelrotation=0, labelsize=10)
            # 根据标准答案大小决定刻度个数
            if correct is not None and correct > 1_000_000:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # 调整子图间距
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=0.8)
    plt.savefig(os.path.join(app.root_path, CHART_PATH), dpi=100, bbox_inches='tight')
    plt.close(fig)

    return render_template('results.html', chart_url=url_for('static', filename='summary.png'),
                           message=None, active_group=group)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
