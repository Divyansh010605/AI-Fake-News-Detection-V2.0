# 🛡️ Veritas AI – Fake News Detection & Verification System

AI-powered Fake News Detection system built using DeBERTa v3 Transformer, Flask, and Explainable AI (XAI).

Veritas AI combines deep learning–based classification with real-time fact cross-verification to detect misinformation from raw text or article URLs.

---

## 🚀 Project Overview

Veritas AI is an end-to-end Fake News Detection platform that:

- Trains a Transformer model (DeBERTa v3) on global + Indian news datasets
- Deploys the trained model using a Flask web application
- Provides explainable AI outputs (top influential words)
- Cross-checks news with reputable media sources
- Includes secure user authentication

---

## 🧠 Model Architecture

- Base Model: `microsoft/deberta-v3-base`
- Framework: HuggingFace Transformers
- Task: Binary Classification (REAL vs FAKE)
- Max Token Length: 512
- Training Strategy:
  - Balanced dataset sampling
  - 3 epochs
  - FP16 mixed precision training
  - Gradient accumulation
  - Best model checkpoint saving

Training pipeline implemented in:
