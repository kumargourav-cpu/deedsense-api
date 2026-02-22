import io
import os
import re
import json
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import docx
from langdetect import detect, LangDetectException
from rapidfuzz import fuzz

# -----------------------------
# Config
# -----------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))

# OCR lang packs installed in Dockerfile: eng + ara
# Example: "eng" or "eng+ara"
OCR_LANG = os.getenv("OCR_LANG", "eng+ara")

PDF_DPI = int(os.getenv("PDF_DPI", "220"))
PDF_OCR_PAGES_LIMIT = int(os.getenv("PDF_OCR_PAGES_LIMIT", "15"))

# Safety controls
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "250000"))  # 250k chars
HISTORY_MAX_ITEMS = int(os.getenv("HISTORY_MAX_ITEMS", "50"))

# In-memory history (no login)
HISTORY: List[Dict[str, Any]] = []

app = FastAPI(title="DeedSense API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def enforce_file_size(file_bytes: bytes):
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB}MB.")

def clip_text(text: str) -> str:
    text = (text or "").strip()
    if len(text) > MAX_TEXT_CHARS:
        return text[:MAX_TEXT_CHARS] + "\n\n[...truncated for safety...]"
    return text

def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def safe_detect_lang(text: str) -> str:
    try:
        # langdetect needs some length
        if not text or len(text.strip()) < 30:
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# -----------------------------
# Extraction
# -----------------------------
def extract_text_from_txt(file_bytes: bytes) -> str:
    enforce_file_size(file_bytes)
    try:
        return file_bytes.decode("utf-8").strip()
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore").strip()

def extract_text_from_docx(file_bytes: bytes) -> str:
    enforce_file_size(file_bytes)
    d = docx.Document(io.BytesIO(file_bytes))
    parts = []
    for p in d.paragraphs:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts).strip()

def extract_text_from_image(file_bytes: bytes) -> str:
    enforce_file_size(file_bytes)
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return pytesseract.image_to_string(img, lang=OCR_LANG).strip()

def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    1) Try normal text extraction via pypdf
    2) If too little text, OCR scan pages using pdf2image + tesseract
    """
    enforce_file_size(file_bytes)
    meta: Dict[str, Any] = {
        "used_ocr": False,
        "pages_ocrd": 0,
        "pages_total": None,
        "reason": None,
    }

    extracted = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        meta["pages_total"] = len(reader.pages)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        extracted = "\n".join(parts).strip()
    except Exception as e:
        meta["reason"] = f"pypdf_failed: {str(e)}"
        extracted = ""

    # If extracted text is tiny, assume scanned PDF
    if len(normalize_ws(extracted)) < 60:
        meta["used_ocr"] = True
        meta["reason"] = meta["reason"] or "low_text_content"
        images = convert_from_bytes(file_bytes, dpi=PDF_DPI)
        ocr_parts = []
        limit = min(len(images), PDF_OCR_PAGES_LIMIT)
        for i in range(limit):
            ocr_parts.append(pytesseract.image_to_string(images[i], lang=OCR_LANG))
        meta["pages_ocrd"] = limit
        extracted = "\n".join(ocr_parts).strip()

    return extracted, meta

def extract_text_by_file(filename: str, content_type: str, content: bytes) -> Tuple[str, Dict[str, Any]]:
    name = (filename or "").lower()
    ctype = (content_type or "").lower()

    meta: Dict[str, Any] = {
        "filename": filename,
        "content_type": content_type,
        "mode": None,
        "ocr": False,
    }

    if name.endswith(".pdf") or "pdf" in ctype:
        meta["mode"] = "pdf"
        text, pdfmeta = extract_text_from_pdf(content)
        meta.update(pdfmeta)
        meta["ocr"] = bool(meta.get("used_ocr"))
        return clip_text(text), meta

    if name.endswith(".docx") or "officedocument.wordprocessingml.document" in ctype:
        meta["mode"] = "docx"
        return clip_text(extract_text_from_docx(content)), meta

    if name.endswith(".txt") or "text/plain" in ctype:
        meta["mode"] = "txt"
        return clip_text(extract_text_from_txt(content)), meta

    if any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) or "image/" in ctype:
        meta["mode"] = "image"
        meta["ocr"] = True
        return clip_text(extract_text_from_image(content)), meta

    raise HTTPException(status_code=415, detail="Unsupported file type. Use PDF, DOCX, TXT, PNG, JPG, JPEG, WEBP.")


# -----------------------------
# Enterprise-grade heuristic scan (no OpenAI required)
# -----------------------------
class ScanIn(BaseModel):
    text: str
    language: Optional[str] = None  # UI can send chosen language code
    context: Optional[Dict[str, Any]] = None

def risk_engine(text: str) -> Dict[str, Any]:
    """
    Research-backed-ish heuristic engine (MVP):
    - Detect persuasion / manipulation patterns
    - Flag missing due diligence items
    - Highlight financial risk claims
    - Generate structured report + scores + charts data
    """
    raw = text or ""
    t = raw.lower()
    clean = normalize_ws(raw)

    signals: List[Dict[str, Any]] = []
    score = 0

    def add_signal(key: str, title: str, severity: str, points: int, evidence: List[str], why: str, what_to_do: str):
        nonlocal score
        score += points
        signals.append({
            "key": key,
            "title": title,
            "severity": severity,
            "points": points,
            "evidence": evidence[:5],
            "why_it_matters": why,
            "what_to_verify": what_to_do,
        })

    def find_evidence(phrases: List[str]) -> List[str]:
        ev = []
        for p in phrases:
            if p in t:
                ev.append(p)
        return ev

    # --- Manipulation / persuasion patterns
    urgency = ["limited time", "today only", "last chance", "offer ends", "deadline", "book now", "reserve now", "act fast"]
    scarcity = ["last unit", "only unit", "few left", "limited units", "sold out soon"]
    certainty = ["guaranteed", "assured", "risk-free", "no risk", "100%"]
    social_proof = ["everyone is buying", "high demand", "many investors", "hot deal"]
    authority = ["government approved", "officially endorsed", "trusted by", "award-winning"]  # generic
    pressure = ["no questions asked", "don't miss", "final call", "just transfer", "pay now"]

    if find_evidence(urgency):
        add_signal(
            "urgency_pressure",
            "Urgency pressure language",
            "medium",
            12,
            find_evidence(urgency),
            "Urgency cues can push investors to decide before verifying documents.",
            "Ask for official documents: SPA, title deed (or Oqood), escrow account proof, payment schedule, cancellation terms."
        )
    if find_evidence(scarcity):
        add_signal(
            "scarcity",
            "Scarcity / FOMO cues",
            "medium",
            10,
            find_evidence(scarcity),
            "Scarcity cues are common in sales scripts and can inflate perceived value.",
            "Verify availability via official inventory / developer confirmation and compare similar units’ pricing."
        )
    if find_evidence(certainty):
        add_signal(
            "guarantees",
            "Guarantees / 'risk-free' claims",
            "high",
            18,
            find_evidence(certainty),
            "Strong guarantees in property investing are often misleading unless backed by enforceable contracts.",
            "Request the written guarantee clause, counterparty identity, enforcement jurisdiction, and exit terms."
        )
    if find_evidence(pressure):
        add_signal(
            "high_pressure",
            "High-pressure persuasion cues",
            "high",
            16,
            find_evidence(pressure),
            "Pressure language correlates with reduced verification and higher regret decisions.",
            "Slow down: verify escrow, developer registration, RERA/land department references, and all fees."
        )

    # --- Financial risk claims
    roi_phrases = ["roi", "returns", "return", "yield", "rental guarantee", "fixed income", "profit"]
    if any(p in t for p in roi_phrases):
        add_signal(
            "roi_claims",
            "ROI / returns language detected",
            "medium",
            10,
            [p for p in roi_phrases if p in t][:5],
            "ROI claims need assumptions (occupancy, service charges, handover timeline, taxes).",
            "Ask for a full rental comp set, service charge estimates, handover date clauses, and vacancy assumptions."
        )

    # --- Missing due diligence checklist
    checklist = [
        ("escrow", ["escrow", "trust account"], "Escrow / trust account proof"),
        ("title_deed", ["title deed", "oqood"], "Title deed / Oqood reference"),
        ("developer", ["developer", "master developer"], "Developer identity / registration"),
        ("fees", ["service charge", "dld", "registration", "commission", "admin fee"], "Fee transparency"),
        ("handover", ["handover", "completion", "delivery"], "Handover date clarity"),
        ("cancellation", ["cancellation", "refund", "termination"], "Cancellation / refund terms"),
    ]

    missing = []
    for key, kws, label in checklist:
        if not any(k in t for k in kws):
            missing.append(label)

    checklist_total = len(checklist)
    checklist_hit = checklist_total - len(missing)
    checklist_completion = round((checklist_hit / checklist_total) * 100)

    if missing:
        add_signal(
            "missing_due_diligence",
            "Missing key due diligence items in the text",
            "medium" if len(missing) <= 2 else "high",
            8 + min(10, len(missing) * 2),
            missing[:5],
            "If basic items aren't mentioned, investors often discover surprises later (fees, timelines, legal terms).",
            "Collect missing items before paying: escrow proof, deed/Oqood, fee sheet, handover schedule, cancellation terms."
        )

    # --- Consistency & contradictions (light)
    # Example: claims of "ready" but mentions "handover"
    if ("ready" in t or "ready to move" in t) and ("handover" in t or "completion" in t):
        add_signal(
            "possible_contradiction",
            "Potential contradiction (ready vs handover wording)",
            "low",
            6,
            ["ready/ready-to-move", "handover/completion"],
            "Contradictory phrasing can confuse status (off-plan vs ready).",
            "Confirm status: ready title deed vs off-plan Oqood; verify handover date in SPA."
        )

    # Score normalization
    score = max(0, min(100, score))
    label = "Low" if score < 20 else "Medium" if score < 50 else "High"
    confidence = round(0.65 + (min(30, len(clean)) / 300) * 0.25, 2)  # simple heuristic

    # Signal density (signals per 1000 chars)
    density = round((len(signals) / max(1, len(clean))) * 1000, 2)

    # Trendline (fake 7-day trend for chart — UI can show it)
    # You can later replace with real history aggregation.
    trend = []
    base = score
    for i in range(7):
        v = max(0, min(100, int(base + (i - 3) * 2)))
        trend.append({"day": f"D-{6-i}", "risk": v})

    return {
        "risk_score": score,
        "risk_label": label,
        "confidence": confidence,
        "language_detected": safe_detect_lang(clean),
        "summary": {
            "headline": f"{label} risk signal ({score}/100)",
            "one_liner": "This is a risk indicator based on textual patterns. Verify using official documents and due diligence.",
        },
        "signals": signals,
        "checklist": {
            "completion_percent": checklist_completion,
            "missing": missing,
            "present": checklist_hit,
            "total": checklist_total,
        },
        "charts": {
            "signal_density": density,
            "trendline_7d": trend,
            "severity_breakdown": {
                "high": sum(1 for s in signals if s["severity"] == "high"),
                "medium": sum(1 for s in signals if s["severity"] == "medium"),
                "low": sum(1 for s in signals if s["severity"] == "low"),
            },
        },
        "tips": [
            "Ask for escrow/trust account proof before any transfer.",
            "Validate deed/Oqood and developer registration with official sources.",
            "Insist on a full fee sheet (DLD, service charges, admin, commission).",
            "Compare comps: similar units, similar handover timelines, similar service charges.",
        ],
        "timestamp": now_iso(),
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "max_upload_mb": MAX_UPLOAD_MB,
        "ocr_lang": OCR_LANG,
        "pdf_dpi": PDF_DPI,
        "pdf_ocr_pages_limit": PDF_OCR_PAGES_LIMIT,
        "time": now_iso(),
    }

@app.get("/info")
def info():
    return {
        "supported_uploads": ["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
        "notes": [
            "Scanned PDFs are OCR’d using Tesseract (requires Docker build).",
            "If a PDF already contains text, OCR is only used when text extraction is insufficient.",
        ],
        "time": now_iso(),
    }

@app.get("/history")
def history():
    return {"items": HISTORY}

@app.post("/scan")
def scan(payload: ScanIn):
    text = clip_text(payload.text or "")
    if len(text.strip()) < 10:
        raise HTTPException(status_code=422, detail="Not enough text to analyze.")

    result = risk_engine(text)

    item = {
        "id": sha256_text(text + now_iso())[:16],
        "created_at": now_iso(),
        "input_type": "text",
        "filename": None,
        "text_hash": sha256_text(text),
        "result": result,
    }
    HISTORY.insert(0, item)
    del HISTORY[HISTORY_MAX_ITEMS:]

    return {"extracted_text": text, "result": result}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """
    Accepts PDF/DOCX/TXT/PNG/JPG/JPEG/WEBP
    Returns extracted text + metadata
    """
    filename = file.filename or "upload"
    content_type = file.content_type or ""
    content = await file.read()

    enforce_file_size(content)

    text, meta = extract_text_by_file(filename, content_type, content)
    if len(text.strip()) < 10:
        raise HTTPException(status_code=422, detail="No readable text found. If scanned, ensure OCR is enabled and scan is clear.")

    meta["language_detected"] = safe_detect_lang(text)
    meta["text_hash"] = sha256_text(text)

    return {"extracted_text": text, "meta": meta}

@app.post("/extract-and-scan")
async def extract_and_scan(file: UploadFile = File(...)):
    """
    One-shot endpoint: upload -> extract -> scan
    """
    filename = file.filename or "upload"
    content_type = file.content_type or ""
    content = await file.read()

    enforce_file_size(content)

    text, meta = extract_text_by_file(filename, content_type, content)
    text = clip_text(text)

    if len(text.strip()) < 10:
        raise HTTPException(status_code=422, detail="No readable text found. If scanned, ensure OCR is enabled and scan is clear.")

    result = risk_engine(text)

    item = {
        "id": sha256_text(text + now_iso())[:16],
        "created_at": now_iso(),
        "input_type": meta.get("mode") or "file",
        "filename": filename,
        "text_hash": sha256_text(text),
        "meta": meta,
        "result": result,
    }
    HISTORY.insert(0, item)
    del HISTORY[HISTORY_MAX_ITEMS:]

    return {"filename": filename, "meta": meta, "extracted_text": text, "result": result}
