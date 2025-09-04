# ai-log-analysis
AI-Powered Log Analysis using AIOps &amp; Isolation Forest — Compare traditional log monitoring with an AI-driven approach for anomaly detection across INFO, WARNING, ERROR, and CRITICAL logs.


           🛠 AI-Powered Log Analysis for DevOps

This project demonstrates AI-powered log analysis using Python and Machine Learning (Isolation Forest) to perform anomaly detection in application logs. It aims to enhance observability and proactively identify issues before they impact application performance.

🔍 Overview

Traditional log analysis in DevOps often relies on parsing logs and counting errors using scripts. However, this approach has limitations:

Only detects explicit errors; ignores warnings, info, or critical messages.

Cannot identify subtle or unusual patterns (e.g., slow response times).

Requires manual thresholds and hardcoded rules.

By integrating AIOps and machine learning, we can automatically detect anomalous patterns in logs, improving system reliability and reducing mean time to detection (MTTD).

⚡ AIOps Context

AIOps leverages AI/ML to analyze data from observability tools (logs, metrics, traces) to:

Detect anomalies in real-time.

Predict potential failures.

Trigger automated actions or notifications via AI agents.

This implementation focuses on observability, where large volumes of log data are available to identify subtle anomalies.

🎯 Goal

Build a Python script using Isolation Forest for anomaly detection.

Detect unusual patterns across all log levels (INFO, WARNING, ERROR, CRITICAL).

Provide actionable insights to DevOps teams.

🧩 Traditional Log Analysis

Steps:

Read the raw log file (system_logs.txt).

Parse logs using Pandas and regular expressions.

Count error logs within a fixed time window (e.g., 30 seconds).

Trigger notifications if thresholds are exceeded.

Sample Code:

import pandas as pd
from collections import Counter
import re

log_file = "system_logs.txt"
log_entries = []

with open(log_file, "r") as file:
    for line in file:
        match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)", line.strip())
        if match:
            timestamp, level, message = match.groups()
            log_entries.append([timestamp, level, message])

df = pd.DataFrame(log_entries, columns=["timestamp", "level", "message"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

error_counts = Counter(df[df["level"] == "ERROR"]["timestamp"].dt.floor("30S"))
threshold = 3

for time, count in error_counts.items():
    if count > threshold:
        print(f"🚨 Anomaly detected! {count} ERROR logs in 30 seconds at {time}")


Limitations:

Ignores warnings, info, and critical logs.

Requires hard-coded thresholds.

Cannot detect subtle patterns like increasing response times.

🧠 Machine Learning Approach: Isolation Forest

Isolation Forest (iForest) is an unsupervised anomaly detection algorithm.

How it works:

Randomly selects features and split values to build Isolation Trees.

Anomalies are isolated faster (shorter paths).

Normal points require more splits (longer paths).

Anomaly scores are computed:

Close to 1 → anomaly

Close to 0 → normal

Advantages over traditional approach:

Detects anomalies across all log levels.

Learns patterns automatically without hardcoded thresholds.

Can identify unusual trends in metrics like response times.

🛠 Implementation

Python Script Using Scikit-learn

import pandas as pd
from sklearn.ensemble import IsolationForest

# Read log file
log_file_path = "system_logs.txt"
with open(log_file_path, "r") as file:
    logs = file.readlines()

# Parse logs into DataFrame
data = []
for log in logs:
    parts = log.strip().split(" ", 3)
    if len(parts) < 4:
        continue
    timestamp = parts[0] + " " + parts[1]
    level = parts[2]
    message = parts[3]
    data.append([timestamp, level, message])

df = pd.DataFrame(data, columns=["timestamp", "level", "message"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Feature engineering
level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
df["level_score"] = df["level"].map(level_mapping)
df["message_length"] = df["message"].apply(len)

# Isolation Forest Model
model = IsolationForest(contamination=0.1, random_state=42)
df["anomaly"] = model.fit_predict(df[["level_score", "message_length"]])
df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

# Print detected anomalies
anomalies = df[df["is_anomaly"] == "❌ Anomaly"]
print("\n🔍 Detected Anomalies:\n", anomalies)


Features:

level_score → Numeric severity of log level.

message_length → Length of the log message.

contamination=0.1 → ~10% of data expected as anomalies.

Predicts anomalies (-1) vs normal logs (1).

📊 Benefits for DevOps

Proactive detection: Identify performance degradation or errors before impact.

Cross-level detection: Catch anomalies in info, warning, error, and critical logs.

Integration-ready: Can trigger notifications or automate remediation using AI agents.

Scalable: Works with large-scale log datasets from multiple services.

⚙️ Requirements

Python 3.8+

Libraries: pandas, scikit-learn, numpy

Install dependencies:

pip install pandas scikit-learn numpy

🔗 References

Scikit-learn Isolation Forest

AIOps and Observability
