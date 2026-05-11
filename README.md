# ⚡ Sentinel — Real-Time Fraud Detection System

A full-stack AI-powered fraud detection system with a live analyst dashboard, PyTorch MLP scoring engine, and SHAP-style explanations.

**Live Demo:**
- Frontend: [sentinel-fraud-detection.vercel.app](https://sentinel-fraud-detection.vercel.app)
- Backend: [sentinel-fraud-detection.onrender.com](https://sentinel-fraud-detection.onrender.com)

## Features

- **Live Transaction Stream** — auto-generates synthetic transactions every 3 seconds and scores them in real time
- **PyTorch MLP Model** — 30-feature engineered input, trained with BCELoss, served via FastAPI
- **SHAP-style Explanations** — permutation importance shows top 5 risk factors per transaction
- **Analyst Dashboard** — approve, block, or request 3DS authentication with optional notes
- **Risk Gauge** — circular conic-gradient visualization of fraud probability (0–100%)
- **Score Distribution Histogram** — pure CSS, no chart libraries
- **Decision Logging** — all analyst actions saved to `decisions.jsonl` on the backend
- **Offline Fallback** — local scoring kicks in automatically if backend is unreachable

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, CSS Grid, Flexbox |
| Backend | FastAPI, Uvicorn |
| ML Model | PyTorch (CPU), BCELoss, Adam |
| Feature Engineering | NumPy (30-dim vector) |
| Deployment | Vercel (frontend), Render (backend) |

---

## Architecture

```
React Dashboard (Vercel)
  ↓  POST /score  every 3s
FastAPI Backend (Render)
  → engineer_features()  →  NumpyScaler  →  FraudMLP (PyTorch)
  → _explain()  (permutation importance)
  ← { fraud_probability, prediction, top_features }

Analyst action
  ↓  POST /decision
FastAPI  →  decisions.jsonl  (append-only audit log)
```

---

## Model Details

| Property | Value |
|---|---|
| Architecture | MLP: 30 → 128 → 64 → 32 → 1 |
| Loss | BCELoss |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Regularization | BatchNorm + Dropout (0.3, 0.2) |
| Input features | 30 engineered from 7 raw fields |
| Training data | 20,000 synthetic samples (15% fraud rate) |
| Explanation method | Permutation importance (per-feature Δ probability) |

### Input Features (7 raw → 30 engineered)

| Raw Field | Engineered Features |
|---|---|
| TransactionAmt | log1p, normalised, is_high (>$500), is_very_high (>$2000) |
| Hour | normalised, is_night (10pm–6am), sin/cos encoding |
| Velocity | normalised, is_high (>7/hr), velocity × amount interaction |
| Merchant | one-hot (10 classes) |
| CardType | one-hot (4 classes) |
| Country | one-hot top-5 + "other" flag |

---

## API Reference

### `POST /score`

**Request:**
```json
{
  "TransactionID": "TXN-A7B2C9D4",
  "TransactionAmt": 1234.56,
  "Merchant": "Amazon",
  "CardType": "Visa",
  "Country": "US",
  "Hour": 14,
  "Velocity": 3,
  "timestamp": "2026-05-11T19:09:00Z"
}
```

**Response:**
```json
{
  "TransactionID": "TXN-A7B2C9D4",
  "fraud_probability": 0.73,
  "prediction": "FRAUD",
  "top_features": [
    { "feature": "Velocity", "impact": 0.35, "value": 8 },
    { "feature": "Amount",   "impact": 0.28, "value": 1234.56 },
    { "feature": "Hour",     "impact": 0.20, "value": 14 },
    { "feature": "Country",  "impact": 0.12, "value": "US" },
    { "feature": "Merchant", "impact": 0.05, "value": "Amazon" }
  ]
}
```

### `POST /decision`

```json
{
  "transaction_id": "TXN-A7B2C9D4",
  "action": "BLOCK",
  "note": "High velocity + unusual hour",
  "modelPrediction": "FRAUD",
  "modelScore": 0.73,
  "timestamp": "2026-05-11T19:10:00Z"
}
```

### `GET /health`

Returns server status and model path.

---

## Running Locally

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

On first run the model auto-trains (~30 seconds) and saves `fraud_model.pth` + `scaler.npz`. Subsequent starts load instantly.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The dashboard connects to `localhost:8000` by default.

---

## Project Structure

```
sentinel-fraud-detection/
├── backend/
│   ├── main.py              # FastAPI app + PyTorch model
│   ├── requirements.txt
│   ├── fraud_model.pth      # trained MLP weights
│   └── scaler.npz           # fitted NumpyScaler
├── frontend/
│   └── src/
│       └── App.jsx          # React dashboard
└── README.md
```

---

## Built By

**Mayokun** — Full-stack engineer & ML practitioner  
[github.com/josephtobimayokun](https://github.com/josephtobimayokun)

> Part of the Microlink portfolio — AI-powered software for real-world problems.
