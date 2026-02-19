import os
import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# OpenAI (your code may use OpenAI or any other analyzer)
# If you don't want OpenAI calls, keep analyze_stub() and return that.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# -----------------------------
# Config
# -----------------------------
API_NAME = "DeedSense API"
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()  # Render Postgres uses this exact name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

ENABLE_DB = bool(DATABASE_URL)  # <-- DB is OPTIONAL now


# -----------------------------
# App
# -----------------------------
app = FastAPI(title=API_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS] if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Schemas
# -----------------------------
class AnalyzeRequest(BaseModel):
    text: str
    meta: Optional[Dict[str, Any]] = None


class AnalyzeResponse(BaseModel):
    summary: str
    risk_score: float
    confidence: float
    signals: list
    details: str
    scores: Optional[Dict[str, float]] = None


# -----------------------------
# Helpers
# -----------------------------
def analyze_stub(text: str) -> AnalyzeResponse:
    """
    No-DB, no-OpenAI fallback so your product ALWAYS works.
    Replace this with your real scoring logic if needed.
    """
    t = text.lower()
    signals = []

    def flag(title, severity="medium"):
        signals.append({"title": title, "severity": severity})

    urgency_words = ["today", "tonight", "last chance", "only now", "hurry", "urgent", "limited time", "final"]
    money_words = ["deposit", "transfer", "pay now", "cash", "bank", "crypto", "wire"]
    vague_words = ["guaranteed", "sure profit", "no risk", "100%"]

    urgency_hits = sum(1 for w in urgency_words if w in t)
    money_hits = sum(1 for w in money_words if w in t)
    vague_hits = sum(1 for w in vague_words if w in t)

    if urgency_hits:
        flag("Urgency / pressure language detected", "high" if urgency_hits >= 2 else "medium")
    if money_hits:
        flag("Payment pressure / transfer language detected", "high" if money_hits >= 2 else "medium")
    if vague_hits:
        flag("Vague guarantee-style claims detected", "medium")

    length_score = min(len(text) / 1200.0, 1.0)  # more text = more confidence
    confidence = 0.45 + 0.45 * length_score
    risk_score = min(0.15 + 0.25 * urgency_hits + 0.20 * money_hits + 0.10 * vague_hits, 0.95)

    summary = (
        "DeedSense flagged potential manipulation/risk patterns. "
        "Review the highlighted signals and validate terms using official documents, escrow/payment proof, "
        "and independent legal due diligence."
    )

    details = (
        "This is an MVP risk signal based on patterns commonly seen in high-pressure property pitches. "
        "If the message includes deposit requests, urgency timelines, or guarantee-style claims, treat it as higher risk "
        "until confirmed by developer documentation and proper contracts."
    )

    scores = {
        "overall": float(risk_score),
        "urgency": float(min(urgency_hits / 3.0, 1.0)),
        "payment_pressure": float(min(money_hits / 3.0, 1.0)),
        "vagueness": float(min(vague_hits / 3.0, 1.0)),
    }

    return AnalyzeResponse(
        summary=summary,
        risk_score=float(risk_score),
        confidence=float(min(max(confidence, 0.0), 1.0)),
        signals=signals,
        details=details,
        scores=scores,
    )


def analyze_with_openai(text: str) -> AnalyzeResponse:
    if OpenAI is None:
        return analyze_stub(text)

    if not OPENAI_API_KEY:
        # If you haven't added billing yet, OpenAI calls may fail.
        # Still return stub so your app works.
        return analyze_stub(text)

    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "You are DeedSense, a trust and manipulation risk scanner for property investors. "
        "Analyze the text (listing/broker message/payment terms) and return strict JSON with:\n"
        "{summary: string, risk_score: number 0..1, confidence: number 0..1, "
        "signals: [{title: string, severity: 'low'|'medium'|'high'}], details: string, "
        "scores: {overall:number, urgency:number, vagueness:number, payment_pressure:number}}\n"
        "Be conservative, avoid legal advice, focus on risk signals and questions to ask."
    )

    user = f"TEXT:\n{text}"

    # Use Chat Completions to avoid the 'responses' attribute error.
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content or ""

    # Extract JSON safely
    json_str = content.strip()
    if json_str.startswith("```"):
        json_str = json_str.strip("`")
        json_str = json_str.replace("json", "", 1).strip()

    try:
        obj = json.loads(json_str)
    except Exception:
        # fall back to stub on parse failure
        return analyze_stub(text)

    # Normalize
    return AnalyzeResponse(
        summary=obj.get("summary") or "Analysis complete.",
        risk_score=float(obj.get("risk_score", 0.35)),
        confidence=float(obj.get("confidence", 0.6)),
        signals=obj.get("signals") or [],
        details=obj.get("details") or "",
        scores=obj.get("scores"),
    )


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "db_enabled": ENABLE_DB}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    text = (req.text or "").strip()
    if len(text) < 10:
        raise HTTPException(status_code=400, detail="Please provide more text to analyze.")

    # IMPORTANT: DB is optional. Don't block scans if DATABASE_URL is missing.
    # If you want to save scans later, you can implement DB saving inside `if ENABLE_DB:`
    try:
        result = analyze_with_openai(text)
    except Exception as e:
        # Always return helpful error
        raise HTTPException(status_code=500, detail=f"Analyze failed: {str(e)}")

    return result.model_dump()
