"""
Fraud Detection FastAPI Backend
================================
Run with: uvicorn main:app --reload --port 8000

Dependencies:
    pip install fastapi uvicorn torch numpy joblib

On first run, if no model/scaler files exist, the app auto-trains a
synthetic MLP so the frontend works out of the box.
"""

import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "fraud_model.pth"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DECISIONS_PATH = BASE_DIR / "decisions.jsonl"

# ── Feature constants ──────────────────────────────────────────────────────────
MERCHANTS = ["Amazon", "Walmart", "Target", "Best Buy", "Apple Store",
             "Gas Station", "Restaurant", "Grocery", "Pharmacy", "Online Gaming"]
CARD_TYPES = ["Visa", "Mastercard", "Amex", "Discover"]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU", "JP", "BR", "IN", "MX"]

# High-risk sets used for manual feature engineering signals
HIGH_RISK_MERCHANTS = {"Online Gaming", "Apple Store", "Best Buy"}
HIGH_RISK_COUNTRIES = {"BR", "MX", "IN"}

INPUT_DIM = 30   # total engineered features fed to the model


# ── MLP Model ─────────────────────────────────────────────────────────────────
class FraudMLP(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(tx: dict) -> np.ndarray:
    """
    Convert raw transaction dict → 30-dim float32 numpy vector.

    Dimensions:
      0   : log1p(TransactionAmt)
      1   : TransactionAmt / 5000  (normalised)
      2   : is_high_amount (>500)
      3   : is_very_high_amount (>2000)
      4   : Hour / 23
      5   : is_night_hour (22-6)
      6   : sin(2π·Hour/24)
      7   : cos(2π·Hour/24)
      8   : Velocity / 10
      9   : is_high_velocity (>7)
      10  : Velocity * TransactionAmt / 50000   (interaction)
     11-20: one-hot Merchant (10 classes)
     21-24: one-hot CardType (4 classes)
     25-34: one-hot Country (10 classes) — we use 25-34, but cap at 30 dims
             so Country uses indices 25-29 (first 5 countries only as OHE,
             rest encoded as "other" flag at index 29)
    """
    amt = float(tx.get("TransactionAmt", 100))
    hour = int(tx.get("Hour", 12))
    velocity = float(tx.get("Velocity", 1))
    merchant = tx.get("Merchant", "Amazon")
    card = tx.get("CardType", "Visa")
    country = tx.get("Country", "US")

    feats = np.zeros(INPUT_DIM, dtype=np.float32)

    # Amount features (0-3)
    feats[0] = math.log1p(amt) / math.log1p(5000)
    feats[1] = amt / 5000.0
    feats[2] = float(amt > 500)
    feats[3] = float(amt > 2000)

    # Hour features (4-7)
    feats[4] = hour / 23.0
    feats[5] = float(hour >= 22 or hour <= 6)
    feats[6] = math.sin(2 * math.pi * hour / 24)
    feats[7] = math.cos(2 * math.pi * hour / 24)

    # Velocity features (8-10)
    feats[8] = velocity / 10.0
    feats[9] = float(velocity > 7)
    feats[10] = (velocity * amt) / 50000.0

    # Merchant one-hot (11-20)
    if merchant in MERCHANTS:
        feats[11 + MERCHANTS.index(merchant)] = 1.0

    # CardType one-hot (21-24)
    if card in CARD_TYPES:
        feats[21 + CARD_TYPES.index(card)] = 1.0

    # Country one-hot first 5 + "other" flag (25-29)
    top5_countries = COUNTRIES[:5]   # US, CA, GB, DE, FR
    if country in top5_countries:
        feats[25 + top5_countries.index(country)] = 1.0
    else:
        feats[29] = 1.0

    return feats


# ── Scaler (simple StandardScaler clone, no sklearn dependency at runtime) ────
class NumpyScaler:
    """Lightweight StandardScaler that serialises to/from a plain dict."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path):
        np.savez(str(path), mean=self.mean_, scale=self.scale_)

    @classmethod
    def load(cls, path: Path):
        data = np.load(str(path))
        s = cls()
        s.mean_ = data["mean"]
        s.scale_ = data["scale"]
        return s


# ── Synthetic training data generator ─────────────────────────────────────────
def _generate_synthetic_training_data(n: int = 20_000):
    rows, labels = [], []
    for _ in range(n):
        is_fraud = random.random() < 0.15  # 15 % fraud rate
        if is_fraud:
            amt = random.uniform(200, 5000)
            hour = random.choice(list(range(0, 7)) + list(range(22, 24)))
            velocity = random.randint(6, 10)
            merchant = random.choice(list(HIGH_RISK_MERCHANTS))
            card = random.choice(CARD_TYPES)
            country = random.choice(list(HIGH_RISK_COUNTRIES) + ["US"])
        else:
            amt = random.uniform(10, 300)
            hour = random.randint(8, 21)
            velocity = random.randint(0, 4)
            merchant = random.choice(MERCHANTS)
            card = random.choice(CARD_TYPES)
            country = random.choice(["US", "CA", "GB", "DE", "FR"])

        tx = dict(TransactionAmt=amt, Hour=hour, Velocity=velocity,
                  Merchant=merchant, CardType=card, Country=country)
        rows.append(engineer_features(tx))
        labels.append(float(is_fraud))

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.float32)


# ── Auto-train if no saved model ──────────────────────────────────────────────
def _train_and_save():
    print("⚙️  No saved model found — training synthetic MLP (≈20 s) …")
    X, y = _generate_synthetic_training_data(20_000)
    scaler = NumpyScaler()
    X_scaled = scaler.fit_transform(X)
    scaler.save(SCALER_PATH.with_suffix(".npz"))

    model = FraudMLP(INPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()

    Xt = torch.tensor(X_scaled)
    yt = torch.tensor(y).unsqueeze(1)

    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        out = model(Xt)
        loss = criterion(out, yt)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/30  loss={loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅  Model saved → {MODEL_PATH}")
    return model, scaler


# ── App startup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Fraud Detection API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: FraudMLP = None
_scaler: NumpyScaler = None


@app.on_event("startup")
def load_model():
    global _model, _scaler
    scaler_npz = SCALER_PATH.with_suffix(".npz")

    if MODEL_PATH.exists() and scaler_npz.exists():
        _model = FraudMLP(INPUT_DIM)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _scaler = NumpyScaler.load(scaler_npz)
        print(f"✅  Loaded model from {MODEL_PATH}")
    else:
        _model, _scaler = _train_and_save()

    _model.eval()


# ── Pydantic schemas ───────────────────────────────────────────────────────────
class TransactionIn(BaseModel):
    TransactionID: str
    TransactionAmt: float
    Merchant: str
    CardType: str
    Country: str
    Hour: int
    Velocity: float
    timestamp: str


class FeatureImpact(BaseModel):
    feature: str
    impact: float
    value: object


class ScoreOut(BaseModel):
    TransactionID: str
    fraud_probability: float
    prediction: str
    top_features: List[FeatureImpact]


class DecisionIn(BaseModel):
    transaction_id: str
    action: str
    note: Optional[str] = ""
    modelPrediction: str
    modelScore: float
    timestamp: str


# ── Explanation (simplified permutation importance) ────────────────────────────
_FEATURE_LABELS = [
    ("Amount (log)", "TransactionAmt"),
    ("Amount (norm)", "TransactionAmt"),
    ("High Amount", "TransactionAmt"),
    ("Very High Amount", "TransactionAmt"),
    ("Hour (norm)", "Hour"),
    ("Night Hour", "Hour"),
    ("Hour (sin)", "Hour"),
    ("Hour (cos)", "Hour"),
    ("Velocity (norm)", "Velocity"),
    ("High Velocity", "Velocity"),
    ("Velocity×Amount", "Velocity"),
] + [(f"Merchant={m}", "Merchant") for m in MERCHANTS] \
  + [(f"Card={c}", "CardType") for c in CARD_TYPES] \
  + [(f"Country={c}", "Country") for c in COUNTRIES[:5]] \
  + [("Country=Other", "Country")]


def _explain(feats_raw: np.ndarray, base_prob: float) -> List[dict]:
    """
    Approximate feature impact by zeroing each feature and measuring Δ probability.
    Groups impacts by the 6 semantic features (Amount, Hour, Velocity, Merchant,
    CardType, Country) and returns top-5.
    """
    group_impacts: dict = {}
    base_tensor = torch.tensor(
        _scaler.transform(feats_raw.reshape(1, -1)), dtype=torch.float32
    )

    with torch.no_grad():
        for i, (_, group) in enumerate(_FEATURE_LABELS[:INPUT_DIM]):
            perturbed = feats_raw.copy()
            perturbed[i] = 0.0
            p_tensor = torch.tensor(
                _scaler.transform(perturbed.reshape(1, -1)), dtype=torch.float32
            )
            perturbed_prob = _model(p_tensor).item()
            delta = abs(base_prob - perturbed_prob)
            group_impacts[group] = group_impacts.get(group, 0.0) + delta

    # Map to display names + raw values
    display_map = {
        "TransactionAmt": ("Amount", float(feats_raw[1] * 5000)),
        "Hour": ("Hour", round(float(feats_raw[4] * 23))),
        "Velocity": ("Velocity", round(float(feats_raw[8] * 10))),
        "Merchant": ("Merchant", _resolve_ohe(feats_raw, 11, MERCHANTS)),
        "CardType": ("CardType", _resolve_ohe(feats_raw, 21, CARD_TYPES)),
        "Country": ("Country", _resolve_country(feats_raw)),
    }

    results = []
    for key, impact in group_impacts.items():
        display_name, raw_val = display_map.get(key, (key, "?"))
        results.append({"feature": display_name, "impact": float(round(impact, 4)), "value": raw_val})

    results.sort(key=lambda x: x["impact"], reverse=True)
    # Normalise impacts to sum ≤ 1
    total = sum(r["impact"] for r in results) or 1
    for r in results:
        r["impact"] = round(r["impact"] / total, 3)

    return results[:5]


def _resolve_ohe(feats: np.ndarray, offset: int, labels: list) -> str:
    for i, lbl in enumerate(labels):
        if feats[offset + i] > 0.5:
            return lbl
    return labels[0]


def _resolve_country(feats: np.ndarray) -> str:
    top5 = COUNTRIES[:5]
    for i, c in enumerate(top5):
        if feats[25 + i] > 0.5:
            return c
    return "Other"


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/score")
def score_transaction(tx: TransactionIn):
    feats = engineer_features(tx.dict())
    scaled = _scaler.transform(feats.reshape(1, -1))
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = float(_model(tensor).item())

    prediction = "FRAUD" if prob > 0.5 else "LEGITIMATE"
    top_features = _explain(feats, prob)

    return ScoreOut(
        TransactionID=tx.TransactionID,
        fraud_probability=round(prob, 4),
        prediction=prediction,
        top_features=top_features,
    )
    clean_features = [
        {
            "feature": f["feature"],
            "impact": float(f["impact"]),
            "value": float(f["value"]) if isinstance(f["value"], (np.floating, np.integer)) else f["value"]
        }
        for f in top_features
    ]
    return {
        "TransactionID": tx.TransactionID,
        "fraud_probability": prob,
        "prediction": prediction,
        "top_features": clean_features,
    }


@app.post("/decision")
def log_decision(decision: DecisionIn):
    record = decision.dict()
    record["server_timestamp"] = datetime.utcnow().isoformat()
    with open(DECISIONS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "ok", "logged": record}


@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH), "timestamp": datetime.utcnow().isoformat()}
