# 🛡️ NetGuard — Cybersecurity ML Portfolio

**Student:** Emmanuel Abolade | C00288657  
**Institution:** South East Technological University, Carlow  
**Module:** Data Science & Machine Learning 2  
**Lecturer:** Greg Doyle  
**Submission:** 17 April 2026

---

## Overview

**NetGuard** applies four machine learning algorithms to real-world cybersecurity problems,
extending the [News→Signal](https://github.com/EmmanuelAbolade/news2signal-portfolio)
financial sentiment portfolio from Semester 1.

Live dashboard: [netguard.streamlit.app](https://share.streamlit.io)

---

## Projects

| # | Algorithm | Project | Dataset |
|---|---|---|---|
| 01 | Support Vector Machine | Email Spam Detection | SMS Spam Collection (UCI) |
| 02 | K-Nearest Neighbours | Network Intrusion Detection | NSL-KDD |
| 03 | K-Means Clustering | Network Traffic Clustering | NSL-KDD |
| 04 | Artificial Neural Network | Cyberattack Classification | NSL-KDD |

---

## Repository Structure

```
NetGuard/
├── 01_SVM_Spam_Detection.ipynb
├── 02_KNN_Network_Intrusion.ipynb
├── 03_KMeans_Network_Clustering.ipynb
├── 04_ANN_Cyberattack_Classification.ipynb
├── app.py                        # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── SMSSpamCollection
│   └── KDDTrain+.txt
└── images/
```

---

## Running Locally

```bash
git clone https://github.com/EmmanuelAbolade/netguard.git
cd netguard
pip install -r requirements.txt
streamlit run app.py
```

---

## Datasets

- **SMS Spam Collection** — Almeida et al. (2011), UCI ML Repository
- **NSL-KDD** — Tavallaee et al. (2009), Canadian Institute for Cybersecurity

---

## Previous Portfolio

[News→Signal](https://github.com/EmmanuelAbolade/news2signal-portfolio) —
Financial news sentiment analysis using Logistic Regression and NLP (Semester 1)

---

*South East Technological University, Carlow — 2026*
