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

METHODS            = ["Not using AI", "Using some AI", "Using refined AI"]
NUM_CSV            = os.path.join("data", "submissions.csv")
TEXT_CSV           = os.path.join("data", "text_responses.csv")
INFO_CSV           = os.path.join("data", "submission_info.csv")

# Sample text answers
SAMPLE_Q4 = [
    "I found Question 4 quite challenging.",
    "Could you clarify the assumptions?",
    "See my attached notes.",
    "N/A"
]
SAMPLE_Q5 = [
    "My opinion is that...",
    "I would recommend further research.",
    "No comment.",
    "N/A"
]

# Sample names
FIRST_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
LAST_NAMES  = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson"]

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Remove old files if present
for path in (NUM_CSV, TEXT_CSV, INFO_CSV):
    if os.path.exists(path):
        os.remove(path)

# 1) Generate numeric submissions.csv (400 students)
with open(NUM_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "method", "answer"])
    for student in range(1, 401):   # 1–400
        for q, correct in CORRECT.items():
            m   = random.choice(METHODS)
            sd  = correct * 0.05
            ans = round(random.normalvariate(correct, sd), 1)
            writer.writerow([q, m, ans])
print(f"✅ Generated {400 * len(CORRECT)} numeric rows in {NUM_CSV}")

# 2) Generate text_responses.csv (80 students)
with open(TEXT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Q4_answer", "Q5_answer"])
    for student in range(1, 81):   # 1–80
        q4 = random.choice(SAMPLE_Q4)
        q5 = random.choice(SAMPLE_Q5)
        writer.writerow([q4, q5])
print(f"✅ Generated 80 text rows in {TEXT_CSV}")

# 3) Generate submission_info.csv (80 students)
with open(INFO_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["section", "team_number", "first_name", "last_name"])
    for student in range(1, 81):   # 1–80
        section     = random.choice(["1", "2", "3", "4", "5", "6"])
        team_number = random.randint(1, 15)
        first_name  = random.choice(FIRST_NAMES)
        last_name   = random.choice(LAST_NAMES)
        writer.writerow([section, team_number, first_name, last_name])
print(f"✅ Generated 80 submission info rows in {INFO_CSV}")
