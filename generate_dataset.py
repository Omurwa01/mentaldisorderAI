"""
generate_dataset.py
--------------------
Generates a realistic DASS-21 inspired dataset for mental disorder diagnosis.
Each of the 21 questions is scored 0-3 (never / sometimes / often / almost always).
Subscales:
  - Depression : Q3, Q5, Q10, Q13, Q16, Q17, Q21
  - Anxiety    : Q2, Q4, Q7, Q9, Q15, Q19, Q20
  - Stress     : Q1, Q6, Q8, Q11, Q12, Q14, Q18

Target classes:
  0 = Normal / Mild
  1 = Moderate / Severe (any subscale in moderate-severe range)

DASS-21 Severity thresholds (raw sum × 2 for DASS-42 equivalent):
  Depression : >=14 → moderate+
  Anxiety    : >=10 → moderate+
  Stress     : >=19 → moderate+
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N = 2000  # total samples
depression_q  = ['Q3','Q5','Q10','Q13','Q16','Q17','Q21']
anxiety_q     = ['Q2','Q4','Q7','Q9','Q15','Q19','Q20']
stress_q      = ['Q1','Q6','Q8','Q11','Q12','Q14','Q18']
all_questions = stress_q + anxiety_q + depression_q  # Q1..Q21

# --- Generate two groups: healthy (60%) and at-risk (40%) ---
n_healthy  = int(N * 0.60)
n_disorder = N - n_healthy

def gen_scores(n, low, high):
    """Random integer scores clamped to [0,3]."""
    raw = np.random.randint(low, high + 1, size=(n, 21))
    return np.clip(raw, 0, 3)

healthy_scores  = gen_scores(n_healthy,  0, 1)   # mostly 0-1 responses
disorder_scores = gen_scores(n_disorder, 1, 3)   # mostly 1-3 responses

scores = np.vstack([healthy_scores, disorder_scores])

df = pd.DataFrame(scores, columns=all_questions)

# Add demographic features (age, gender, sleep_hours, exercise_days_per_week)
ages   = np.random.randint(18, 65, size=N)
gender = np.random.choice(['Male', 'Female', 'Other'], size=N, p=[0.48, 0.48, 0.04])
sleep  = np.round(np.random.normal(6.5, 1.2, size=N).clip(3, 10), 1)
exercise = np.random.randint(0, 8, size=N)

df.insert(0, 'age', ages)
df.insert(1, 'gender', gender)
df.insert(2, 'sleep_hours', sleep)
df.insert(3, 'exercise_days_per_week', exercise)

# Compute subscale totals
df['depression_score'] = df[depression_q].sum(axis=1)
df['anxiety_score']    = df[anxiety_q].sum(axis=1)
df['stress_score']     = df[stress_q].sum(axis=1)

# Assign target (1 if ANY subscale in moderate-severe range)
# DASS-21 thresholds: Depression>=10, Anxiety>=7, Stress>=15  (DASS-21 direct)
df['target'] = (
    (df['depression_score'] >= 10) |
    (df['anxiety_score']    >= 7)  |
    (df['stress_score']     >= 15)
).astype(int)

# Drop intermediate subscale columns (keep only features + target)
df.drop(['depression_score', 'anxiety_score', 'stress_score'], axis=1, inplace=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
os.makedirs("data", exist_ok=True)
df.to_csv("data/mental_disorders.csv", index=False)

print(f"Dataset saved → data/mental_disorders.csv")
print(f"Shape       : {df.shape}")
print(f"Class dist  : {df['target'].value_counts().to_dict()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
