# simulate_data.py
import csv, os, random

# 题目键 & 对应标准答案
CORRECT = {
    "Q1a":   2207682.7,
    "Q1b":    822208.6,
    "Q1c":      5120.3,
    "Q2a":    146640.1,
    "Q2b1":     3330.1,
    "Q2b2":       65.0,
    "Q3a1":  1533691.5,
    "Q3a2":      2297.9,
    "Q3b1":  1768829.7,
    "Q3b2":       14.6
}

METHODS = ["Not using AI", "Using some AI", "Using refined AI"]
CSV_PATH = os.path.join("data", "submissions.csv")

# 删除旧数据
if os.path.exists(CSV_PATH):
    os.remove(CSV_PATH)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "answer", "method"])
    for student in range(1, 401):   # 1–400
        for q, correct in CORRECT.items():
            m   = random.choice(METHODS)
            # 相对标准答案 5% 的正态扰动
            sd  = correct * 0.05
            ans = round(random.normalvariate(correct, sd), 1)
            writer.writerow([q, ans, m])

print(f"✅ 已生成 {400 * len(CORRECT)} 条模拟记录到 {CSV_PATH}")
