# Generative AI-Powered Phishing Email Detection System

## Overview
An intelligent phishing email detection system that combines 
Deep Learning, Generative AI, and Explainable AI to detect 
phishing emails in real time and continuously improve through 
adaptive learning.

## Features
- Real-time phishing email detection using BERT transformer model
- Explainable AI using LIME to highlight influential words
- Adaptive learning pipeline using TinyLlama for synthetic 
  data generation
- Human-in-the-loop feedback mechanism for continuous improvement
- Automated model retraining and deployment pipeline
- Interactive dashboard showing predictions and explanations

## Tech Stack
- **Frontend:** React.js, Vite, CSS
- **Backend:** Python, Flask, Docker
- **AI/ML:** BERT, LIME, TinyLlama, PyTorch, 
  HuggingFace Transformers
- **Database:** Firebase Firestore
- **Email Integration:** Gmail API, Google Cloud Pub/Sub
- **Automation:** Firebase Cloud Functions
- **Hosting:** HuggingFace Spaces
- **Model Storage:** HuggingFace Hub

## System Architecture
The system consists of three major modules:

### 1. Real-Time Phishing Detection
- Gmail API fetches incoming emails via Google Cloud Pub/Sub
- BERT classifies each email as Phishing or Legitimate
- Confidence score determines certainty of prediction
- LIME generates word-level explanations for each prediction
- Results stored in Firebase Firestore and displayed on dashboard

### 2. Adaptive Learning Pipeline
- Low confidence emails flagged for human review
- Users label emails through feedback interface
- Firebase Cloud Function automatically triggers retraining
- TinyLlama generates 5 synthetic similar emails per feedback
- BERT retrains on new data and auto-deploys updated model

### 3. Explainable AI
- LIME perturbs input text 50 times to identify influential words
- Positive weights indicate words pushing towards Phishing
- Negative weights indicate words supporting Legitimate
- Visual word highlights shown on dashboard

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| BERT | 93.64% | 93.85% | 93.35% | 93.60% |
| DistilBERT | 91.92% | 91.93% | 91.68% | 91.81% |
| XLM-RoBERTa | 92.44% | 80.60% | 91.53% | 85.71% |

BERT was selected as the best model with highest accuracy 
and balanced precision-recall across all metrics.

## Dataset
- Training data: Combination of real-world phishing datasets 
  (Nazario Phishing Corpus, Enron Email Dataset)
- Synthetic data: Generated using Vicuna-7B for initial training
- Adaptive data: Generated using TinyLlama during retraining

## Project Structure
```text
phishing-detection-system/
│
├── backend/
│   ├── model_training/                  
│   │   ├── bert_training_mixed.ipynb   #training code of BERT with LIME explanation and confusion matrix
│   │   ├── distilbert_training.ipynb   #training code of DistilBERT with LIME explanation and confusion matrix
│   │   ├── tiny_llama.ipynb            #training code of Tiny Llama 
│   │   ├── VICUNA_Phishing.ipynb       #training and generation code of phishing emails using Vicuna 
│   │   ├── VICUNAS_Safe.ipynb          #training and generation code of legitimate emails using Vicuna 
│   │   └── xlm_training_mixed.ipynb    #training code of XLM-RoBERTa with LIME explanation and confusion matrix
│   │
│   └── phishing-backend/
│       ├── app.py              # Flask backend
│       ├── pipeline.py         # Adaptive learning pipeline
│       ├── Dockerfile          # Container configuration
│       └── requirements.txt    # Python dependencies
├── frontend/
│   └── src/
│       ├── Dashboard.jsx       # Email prediction dashboard
│       ├── Feedback.jsx        # Human review interface
│       └── App.jsx             # Main application
└── cloud-functions/
    └── index.js                # Firebase auto-trigger function

```
## How It Works

### Real-Time Detection Flow
New email arrives in Gmail
↓
Google Cloud Pub/Sub pushes webhook to Flask
↓
BERT classifies as Phishing/Legitimate
↓
LIME explains word-level influences
↓
Results stored in Firebase + shown on dashboard

### Adaptive Learning Flow
Low confidence email flagged for review
↓
User labels email in feedback interface
↓
Firebase Cloud Function triggers pipeline
↓
TinyLlama generates 5 similar emails
↓
BERT retrains on new data
↓
Updated model pushed to HuggingFace Hub
↓
Flask auto-reloads new model

## Deployment
The system is deployed entirely on free cloud platforms:
- **Backend:** HuggingFace Spaces (16GB RAM, free tier)
- **Model:** HuggingFace Hub (version control, free tier)
- **Database:** Firebase Firestore (free tier)
- **Functions:** Firebase Cloud Functions (free tier)
- **Email:** Gmail API (free tier)

 ## Institution
Department of Computer Science and Engineering  
Mar Baselios College of Engineering and Technology (Autonomous)  
Thiruvananthapuram - 695015  
March 2026

## License
This project was developed as part of B.Tech final year project 
at APJ Abdul Kalam Technological University.

