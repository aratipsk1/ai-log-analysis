# ai-log-analysis
AI-Powered Log Analysis using AIOps &amp; Isolation Forest — Compare traditional log monitoring with an AI-driven approach for anomaly detection across INFO, WARNING, ERROR, and CRITICAL logs.
AI powered Log Analysis

           We will develop python script where we will use a Machine Learning algorithm
Which helps us perform Anomaly Detection.

AIOps:

        AIOps is using AI to perform analysis on the data to anticipate(Predict) any strange patterns in the data which we also called as Anomalies and if possible avoid happening of that event  .It  be send notifications to the right people or taking required action using AI agents.All of these needs data.So in devops engineering or SR Engineering the most important implementation of AIOps where AIOps is used is in the space of Observability.Because in the space of observability we can use huge amount of data that comes from metrics,logs and traces. 

Goal:

        We will write python script in the python script we will use machine learning algorithm called Isolation Forest.

Why Log Analysis is Important:

Diagram
   
Proactive Approach—-----AI ---------- Machine Algorithm_______Isolation Forest
                                                                                                        Anomalies —---detect and avoid 


Using Machine learning Algorithm that is Isolation forest AIops that can identify that strange pattern or that anomaly in your application and it can detect and also it can avoid if you integrate that with an AI agent or send notifications to the teams.



Part1: First we will write traditional python script:

I have taken an application log with almost a thousand lines.It has combination of info logging,error logging,warning logging and also some of the critical logs

If the company is not using ELK then usually they will do below task

1.Read a file
2.searealize (expected format)
python libraries like panda
Time \ log \String
  (This is a traditional approach.)
3.Write the for ,while loop and will try to counts errors.Our focus is to count error logs.
4. We will take count of error logs and using if condition we will check if count== is greater or equal to threshold we will say print issues with the log or print there is some problem with the logs.or else smtp and mail libraries and send notification to the companies email address.
This traditional approach below is the Python script:

import pandas as pd
from collections import Counter
import re

# Read log file
log_file = "system_logs.txt"

log_entries = []
with open(log_file, "r") as file:
    for line in file:
        match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)", line.strip())
        if match:
            timestamp, level, message = match.groups()
            log_entries.append([timestamp, level, message])

# Convert to DataFrame
df = pd.DataFrame(log_entries, columns=["timestamp", "level", "message"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Count errors in the last 30 seconds
error_counts = Counter(df[df["level"] == "ERROR"]["timestamp"].dt.floor("30S"))

# Threshold for detecting an anomaly (too many errors in a short time)
threshold = 3

# Detect error spikes
for time, count in error_counts.items():
    if count > threshold:
        print(f"🚨 Anomaly detected! {count} ERROR logs in 30 seconds at {time}")

# Show logs with anomalies
print("\nFull Log Analysis:")
print(df)


Explanation:
Imported panda because in python we first need package module .
Serealizing data or putting data in the right format so we are using panda
We will count the errors for that i am using collections in python where i imported the counter
Aslo I used regular expression .
Requirement of Regular expression is:In our application logging system_log.txt here we are having all verity of logs but sometime applications logging might have a Stack Trace.so we should ignoring all this stack tracing while log analysis for that purpose we used regular expression.





















Now there are so many issues with these scripts.In this script we have not used any Machine learning algorithm.So every time there is a error 
print(f"🚨 Anomaly detected! {count} ERROR logs in 30 seconds at {time}")
                       this particular script print that anomalies detected.May be that error we can ignore,and most important thing is this script also ignores warning messages ,info messages,critical messages.Sometimes your log level might be info , the information might beApplication is leaking the memory. This script will only focus on the Error log level.But sometime log level might be Info,warning,critical.Using AI it will detect strange pattern.
 
Info log pattern-response time-10 ms
                                                      60ms
                                                       5 sec,25 sec
        If we are using AI that will notice this strange pattern.Immediately AI will tell you “there is anomaly data detected,and if you ignore this in future this will go to 60 sec which is one minute and your application would respond back with context deadline or Down state.Thats the reason we have another python script using Machine learning Algorithm.



Part 2:To make it better we will use Machine Learning Algorithm ‘Isolation Forest’

Isolation Forest (iForest) is an unsupervised anomaly detection algorithm.

How It Works
Random Subsampling


Pick a random feature.


Pick a random split value between min and max of that feature.


Tree Building (Isolation Trees / iTrees)


Recursively partition the dataset.


Anomalies (rare and different) tend to end up in leaf nodes with fewer splits (shorter paths).


Normal points need more splits to get isolated.


Scoring


The average path length of a point across many trees is used.


Shorter path length → more likely anomaly.


An anomaly score is computed:


Close to 1 → anomalous


Close to 0 → normal


(Python – Scikit-learn)


import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Read log file
log_file_path = "system_logs.txt"  # Update with your file path if needed
with open(log_file_path, "r") as file:
    logs = file.readlines()

# Parse logs into a structured DataFrame
data = []
for log in logs:
    parts = log.strip().split(" ", 3)  # Ensure the message part is captured fully
    if len(parts) < 4:
        continue  # Skip malformed lines
    timestamp = parts[0] + " " + parts[1]
    level = parts[2]
    message = parts[3]
    data.append([timestamp, level, message])

df = pd.DataFrame(data, columns=["timestamp", "level", "message"])

# Convert timestamp to datetime format for sorting
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Assign numeric scores to log levels
level_mapping = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
df["level_score"] = df["level"].map(level_mapping)

# Add message length as a new feature
df["message_length"] = df["message"].apply(len)

# AI Model for Anomaly Detection (Isolation Forest)
model = IsolationForest(contamination=0.1, random_state=42)  # Lower contamination for better accuracy
df["anomaly"] = model.fit_predict(df[["level_score", "message_length"]])

# Mark anomalies in a readable format
df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

# Print only detected anomalies
anomalies = df[df["is_anomaly"] == "❌ Anomaly"]
print("\n🔍 **Detected Anomalies:**\n", anomalies)

     code does
Read raw log file (system_logs.txt).


Parse logs → Extract timestamp, level, and message.


Feature engineering:


level_score (numeric severity of log level).


message_length (length of the log message).


Train Isolation Forest on [level_score, message_length].


Predict anomalies → -1 = anomaly, 1 = normal.


Marks anomalies clearly (❌ Anomaly vs ✅ Normal).
model = IsolationForest(contamination=0.1, random_state=42)
IsolationForest: A machine learning algorithm for anomaly detection.
contamination=0.1: Tells the model we expect ~10% of the data to be anomalies (outliers).random_state=42: Fixes randomness for reproducibility (so results are the same every run).

Prints detected anomalies only.

Fit the model and predict anomalies
df["anomaly"] = model.fit_predict(df[["level_score", "message_length"]])

df[["level_score", "message_length"]] → Only these two features are used:


level_score (numeric severity of the log level).


message_length (length of the log message).


fit_predict():


Fits the model to this feature space.


Predicts each row as either:


1 → normal


-1 → anomaly


The result is stored in a new column called anomaly.



. Make results more human-readable
df["is_anomaly"] = df["anomaly"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

Uses .apply() with a lambda function:


If value is -1 → mark as "❌ Anomaly"


If value is 1 → mark as "✅ Normal"


Now you can instantly tell which logs are suspicious.


Now perform analysis using this:






So first will run Scikit-learn is a open-source machine learning library for the Python.



Now we installed  Scikit-learn 




