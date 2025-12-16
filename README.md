# Learning-Based Anomaly Detection in Multivariate Time-Series Data

**Agentic Analytics Copilot with Hybrid Statistical and LSTM Models**

**Author:** Rayan Belaam  
**Courses:**  
- ECE 5831 – Neural Networks & Pattern Recognition  
- CIS 579 – Artificial Intelligence  
**Semester:** Fall 2025  

---

## Project Overview

This project presents an **Agentic Analytics Copilot** designed to autonomously monitor multivariate time-series data, detect anomalies, and recommend actions. The system combines classical statistical modeling with neural network–based forecasting to improve anomaly detection robustness and adaptability.

A hybrid approach is used:
- **STL decomposition** captures trend and seasonality structure.
- **LSTM neural networks** learn temporal dependencies and multivariate behavior.
- **Hybrid anomaly scoring** combines both perspectives.
- An **agentic decision layer** suggests next actions and incorporates user feedback.

The project emphasizes **pattern recognition**, **generalization**, and **learning-based anomaly detection** rather than threshold-based monitoring.

---

## System Components

- **Statistical Model:** STL decomposition with robust z-score detection  
- **Neural Model:** LSTM forecasting with pretraining and fine-tuning  
- **Learning Strategy:** Transfer learning across multiple synthetic sites  
- **Agentic Logic:** Rule-based playbook with memory (accept/reject feedback)  
- **Implementation:** Python, TensorFlow/Keras, Statsmodels  

---

## Dataset

Due to privacy constraints of real analytics data, a **synthetic multi-site dataset** was generated.

- 50 independent sites  
- 180 days per site  
- Multivariate KPIs:
  - Sessions
  - Conversion Rate
  - Revenue
- Injected anomalies with seasonality and noise

📂 **Dataset:**  
👉 [Download dataset](https://drive.google.com/drive/folders/1rxKORQmE02-kcEzqTmmt13S50SNd87P9)

📄 **Data generation script:**  
`data_generation_multi.py`

---

## Training Pipeline

1. **Pretraining:**  
   LSTM trained on all sites to learn general KPI dynamics.

2. **Fine-Tuning:**  
   Pretrained weights adapted to a single target site (site 0).

3. **Inference:**  
   Prediction error converted to anomaly scores and fused with STL output.

---

## Results Summary

- Stable convergence during LSTM pretraining
- Controlled adaptation during fine-tuning
- No observed overfitting (training vs validation loss)
- Improved anomaly sensitivity using hybrid scoring

---

## Project Artifacts

### 🎥 Pre-Recorded Presentation
👉 [Presentation Video](https://youtu.be/taQCiOL8EcU)

### 📊 Presentation Slides
👉 [Slides (PowerPoint)](https://drive.google.com/drive/folders/1uKtc_BmY0dS8mPqWWQQIBfNmSTEkvTT7)

### 📄 Final Report
👉 [Final Project Report](https://drive.google.com/drive/folders/1J3izmTs19dJCu87y5FQv4dNyGqD1JWf1)

### 📈 Google Drive
👉 [Google Drive link](https://drive.google.com/drive/folders/1I1qdhajNFWzUdsZHa_lGctwhLTuxBhhy)

### ▶️ Demo Video
👉 [System Demo Video](https://youtu.be/taQCiOL8EcU?t=449)

---

## How to Run

```bash
# Pretrain LSTM
python train_lstm_base.py

# Fine-tune on target site
python finetune_lstm_site.py

# Run the agentic analytics copilot
python main.py

