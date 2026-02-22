import os
import io
import re
import math
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pypdf import PdfReader
import docx  # python-docx
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # stable results

# -----------------------------
# Config
# -----------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))
OCR_LANG = os.getenv("OCR_LANG", "eng+ara")  # example: "eng" or "eng+ara"
MAX_PDF_PAGES_OCR = int(os.getenv("MAX_PDF_PAGES_OCR", "15"))
PDF_DPI = int(os.getenv("PDF_DPI", "220"))

app = FastAPI(title="DeedSense API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class AnalyzeIn(BaseModel):
    text: str
    preferred_language: Optional[str] = None  # e.g. "en", "ar", "hi", "fr"
    context: Optional[Dict[str, Any]] = None  # future use (deal metadata)

class ExtractOut(BaseModel):
    filename: str
    input_type: str
    detected_language: str
    extracted_text: str

class ScanOut(BaseModel):
    filename: Optional[str]
    input_type: str
    detected_language: str
    preferred_language: str
    extracted_text: str
    analysis: Dict[str, Any]
    charts: Dict[str, Any]
    checklist: Dict[str, Any]
    signals: List[Dict[str, Any]]
    meta: Dict[str, Any]

# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def enforce_file_size(b: bytes):
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(b) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB}MB.")

def safe_decode_text(b: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return b.decode(enc, errors="ignore")
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")

def detect_lang(text: str) -> str:
    t = (text or "").strip()
    if len(t) < 20:
        return "en"
    try:
        code = detect(t)
        return code or "en"
    except Exception:
        return "en"

def normalize_whitespace(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# -----------------------------
# Extraction
# -----------------------------
def extract_pdf_text(pdf_bytes: bytes) -> Tuple[str, str]:
    """
    1) Try pypdf extraction.
    2) If too little text, OCR fallback (scanned PDF) via pdf2image + pytesseract.
    Returns (text, mode) where mode is 'pdf-text' or 'pdf-ocr'
    """
    enforce_file_size(pdf_bytes)

    extracted = ""
    mode = "pdf-text"

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for p in reader.pages:
            parts.append((p.extract_text() or "").strip())
        extracted = "\n".join([x for x in parts if x]).strip()
    except Exception:
        extracted = ""

    # OCR fallback if not enough text
    if len(extracted) < 80:
        mode = "pdf-ocr"
        images = convert_from_bytes(pdf_bytes, dpi=PDF_DPI)
        ocr_parts = []
        for img in images[:MAX_PDF_PAGES_OCR]:
            ocr_parts.append(pytesseract.image_to_string(img, lang=OCR_LANG))
        extracted = "\n".join(ocr_parts)

    return normalize_whitespace(extracted), mode

def extract_docx_text(docx_bytes: bytes) -> str:
    enforce_file_size(docx_bytes)
    d = docx.Document(io.BytesIO(docx_bytes))
    text = "\n".join([p.text for p in d.paragraphs if p.text])
    return normalize_whitespace(text)

def extract_image_text(img_bytes: bytes) -> str:
    enforce_file_size(img_bytes)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    text = pytesseract.image_to_string(img, lang=OCR_LANG)
    return normalize_whitespace(text)

def extract_txt_text(txt_bytes: bytes) -> str:
    enforce_file_size(txt_bytes)
    return normalize_whitespace(safe_decode_text(txt_bytes))

def infer_type(filename: str, content_type: str) -> str:
    n = (filename or "").lower()
    ct = (content_type or "").lower()
    if n.endswith(".pdf") or "pdf" in ct:
        return "pdf"
    if n.endswith(".docx") or "wordprocessingml.document" in ct:
        return "docx"
    if n.endswith(".txt") or "text/plain" in ct:
        return "txt"
    if any(n.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")) or ct.startswith("image/"):
        return "image"
    return "unknown"

# -----------------------------
# Analysis Engine (non-LLM, deterministic)
# IMPORTANT: This is NOT "research-backed AI" – it’s rules + scoring.
# It’s designed to be consistent, explainable, and enterprise-style structured.
# -----------------------------
RISK_CATEGORIES = [
    ("manipulation", "Manipulation & pressure"),
    ("financial", "Financial claims & terms"),
    ("legal", "Legal / ownership clarity"),
    ("documentation", "Missing documents / proof"),
    ("quality", "Listing quality & ambiguity"),
]

SIGNALS = [
    # Manipulation / pressure
    ("limited time", "manipulation", 12, "Urgency pressure: 'limited time'"),
    ("last unit", "manipulation", 12, "Scarcity pressure: 'last unit'"),
    ("book now", "manipulation", 8, "CTA pressure: 'book now'"),
    ("today only", "manipulation", 10, "Deadline pressure: 'today only'"),
    ("guaranteed", "financial", 14, "Guarantee language: 'guaranteed'"),
    ("guaranteed returns", "financial", 18, "Suspicious ROI guarantee"),
    ("risk-free", "financial", 12, "Unrealistic safety claim: 'risk-free'"),
    ("no risk", "financial", 12, "Unrealistic safety claim: 'no risk'"),
    ("0% commission", "quality", 6, "Marketing bait: '0% commission'"),
    ("exclusive", "manipulation", 5, "Exclusivity framing: 'exclusive'"),

    # Financial / terms
    ("payment plan", "financial", 6, "Mentions payment plan"),
    ("roi", "financial", 8, "Mentions ROI"),
    ("rental", "financial", 5, "Mentions rental / rent"),
    ("service charge", "financial", 8, "Mentions service charges"),
    ("handover", "financial", 7, "Mentions handover timeline"),

    # Legal / ownership clarity
    ("title deed", "legal", 12, "Mentions title deed"),
    ("oqood", "legal", 10, "Mentions Oqood (UAE off-plan)"),
    ("escrow", "legal", 10, "Mentions escrow"),
    ("noc", "legal", 6, "Mentions NOC"),
    ("freehold", "legal", 6, "Mentions freehold"),
    ("leasehold", "legal", 6, "Mentions leasehold"),

    # Documentation
    ("passport", "documentation", 7, "Requests passport copy (privacy risk depending on context)"),
    ("deposit", "documentation", 7, "Mentions deposit"),
    ("invoice", "documentation", 8, "Mentions invoice"),
    ("receipt", "documentation", 9, "Mentions receipt / proof"),
]

CHECKLIST_ITEMS = [
    ("Price stated clearly", [r"\b(aed|usd|eur|inr)\b", r"\b\d{3,}\b", r"\bprice\b"]),
    ("Unit details present", [r"\b(bed|bhk|sqm|sq\.? ?ft|bua|plot)\b", r"\bunit\b", r"\bsize\b"]),
    ("Payment terms mentioned", [r"\b(payment plan|installment|deposit|down payment)\b"]),
    ("Handover / readiness mentioned", [r"\b(handover|ready|completion)\b"]),
    ("Legal ownership mention", [r"\b(title deed|oqood|escrow|noc)\b"]),
    ("Fees/charges mention", [r"\b(service charge|dld|registration|agency fee)\b"]),
    ("Location/community present", [r"\b(jvc|dubai|abu dhabi|marina|downtown|business bay)\b", r"\blocation\b"]),
    ("Developer/builder named", [r"\bdeveloper\b", r"\b(damac|emaar|nakheel|sobha|meraas)\b"]),
]

def score_text(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    words = re.findall(r"\w+", t)
    wc = len(words)
    if wc == 0:
        wc = 1

    # signal detection
    signals_out: List[Dict[str, Any]] = []
    cat_scores = {k: 0 for k, _ in RISK_CATEGORIES}

    for phrase, cat, weight, label in SIGNALS:
        if phrase in t:
            cat_scores[cat] += weight
            signals_out.append({
                "category": cat,
                "weight": weight,
                "label": label,
                "evidence": phrase,
            })

    # ambiguity penalties (enterprise-style)
    ambiguity = 0
    if len(text) < 280:
        ambiguity += 10
    if re.search(r"\b(call|dm|whatsapp)\b", t) and not re.search(r"\bprice\b|\b(aed|usd|eur|inr)\b", t):
        ambiguity += 12
    if re.search(r"\bguaranteed\b", t) and not re.search(r"\bterms\b|\bconditions\b", t):
        ambiguity += 8

    cat_scores["quality"] += ambiguity

    # normalize into 0-100
    raw_total = sum(cat_scores.values())
    # dampening so it doesn't blow up
    risk_score = int(min(100, round(raw_total * 0.9)))

    if risk_score < 25:
        label = "Low"
    elif risk_score < 55:
        label = "Medium"
    else:
        label = "High"

    # signal density per 1000 words
    signal_density = round((len(signals_out) / wc) * 1000, 2)

    # checklist
    checklist = {}
    completed = 0
    for name, patterns in CHECKLIST_ITEMS:
        ok = False
        for pat in patterns:
            if re.search(pat, t, flags=re.IGNORECASE):
                ok = True
                break
        checklist[name] = ok
        if ok:
            completed += 1

    checklist_completion = round((completed / max(1, len(CHECKLIST_ITEMS))) * 100, 1)

    # confidence heuristic
    # more text + more evidence improves confidence
    confidence = 0.55
    confidence += min(0.25, wc / 2000)
    confidence += min(0.15, len(signals_out) / 20)
    confidence -= 0.10 if ambiguity >= 15 else 0.0
    confidence = float(max(0.35, min(0.92, confidence)))

    # recommendations (structured)
    tips = []
    if not checklist.get("Price stated clearly"):
        tips.append("Request an official price breakdown with currency, unit number, and inclusions/exclusions.")
    if not checklist.get("Fees/charges mention"):
        tips.append("Ask for service charges, DLD/registration fees, agency fee, and any hidden admin charges in writing.")
    if any(s["category"] == "manipulation" for s in signals_out):
        tips.append("Slow the process down. High-pressure language is a red flag — verify documents before paying any deposit.")
    if any(s["evidence"] == "guaranteed returns" for s in signals_out):
        tips.append("Treat ROI guarantees as marketing. Ask for audited rental comps and contract terms before believing projections.")
    if not checklist.get("Legal ownership mention"):
        tips.append("Request proof of ownership/registration (Title Deed / Oqood / escrow details) before transferring funds.")
    if "passport" in t and "receipt" not in t:
        tips.append("Do not share sensitive ID documents without a verified process and clear purpose (privacy risk).")

    if len(tips) < 3:
        tips.append("Always validate claims via official documents, escrow/payment proof, and independent due diligence.")

    # build category breakdown percent
    cat_breakdown = []
    for k, label_k in RISK_CATEGORIES:
        val = int(min(100, cat_scores.get(k, 0)))
        cat_breakdown.append({"key": k, "label": label_k, "value": val})

    # chart structures
    charts = {
        "categoryBreakdown": cat_breakdown,
        "signalDensity": {"value": signal_density, "per": "1000_words"},
        "checklistCompletion": {"value": checklist_completion},
        "trendPoint": {"timestamp": now_iso(), "risk_score": risk_score, "confidence": round(confidence, 2)},
    }

    # executive summary
    executive = {
        "risk_score": risk_score,
        "risk_label": label,
        "confidence": round(confidence, 2),
        "what_it_means": (
            "Low risk means fewer manipulation/ambiguity signals, not a guarantee of safety."
            if label == "Low" else
            "Medium risk suggests meaningful ambiguity or pressure tactics — verify key terms before proceeding."
            if label == "Medium" else
            "High risk indicates strong pressure/guarantee patterns or missing essentials — pause and demand proof."
        ),
    }

    return {
        "executive": executive,
        "signals": signals_out,
        "checklist": checklist,
        "recommendations": tips,
        "charts": charts,
        "meta": {
            "word_count": wc,
            "raw_total": raw_total,
            "timestamp": now_iso(),
            "engine": "rules-v2",
        },
    }

# -----------------------------
# Language output wrapper (UI controls language; API returns detected too)
# -----------------------------
def resolve_preferred_lang(detected: str, preferred: Optional[str]) -> str:
    p = (preferred or "").strip().lower()
    if p:
        return p
    return detected or "en"

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
        "pdf_ocr_pages_limit": MAX_PDF_PAGES_OCR,
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

@app.post("/extract", response_model=ExtractOut)
async def extract(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    content_type = file.content_type or ""
    b = await file.read()
    enforce_file_size(b)

    t = infer_type(filename, content_type)
    if t == "unknown":
        raise HTTPException(status_code=415, detail="Unsupported file type. Use PDF, DOCX, TXT, PNG/JPG/JPEG.")

    extracted = ""
    input_type = t

    try:
        if t == "pdf":
            extracted, mode = extract_pdf_text(b)
            input_type = mode
        elif t == "docx":
            extracted = extract_docx_text(b)
        elif t == "txt":
            extracted = extract_txt_text(b)
        elif t == "image":
            extracted = extract_image_text(b)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    if not extracted or len(extracted) < 15:
        raise HTTPException(status_code=422, detail="No readable text found. For scanned docs, ensure OCR is enabled and scan is clear.")

    detected = detect_lang(extracted)
    return {
        "filename": filename,
        "input_type": input_type,
        "detected_language": detected,
        "extracted_text": extracted,
    }

@app.post("/analyze", response_model=ScanOut)
def analyze(payload: AnalyzeIn):
    text = normalize_whitespace(payload.text or "")
    if len(text) < 20:
        raise HTTPException(status_code=422, detail="Text too short. Paste more content or upload a file.")

    detected = detect_lang(text)
    preferred = resolve_preferred_lang(detected, payload.preferred_language)

    scored = score_text(text)

    return {
        "filename": None,
        "input_type": "text",
        "detected_language": detected,
        "preferred_language": preferred,
        "extracted_text": text,
        "analysis": scored["executive"],
        "charts": scored["charts"],
        "checklist": scored["checklist"],
        "signals": scored["signals"],
        "meta": {
            **scored["meta"],
            "preferred_language": preferred,
            "disclaimer": "Not legal advice. Validate via official documents and due diligence.",
        },
    }

@app.post("/scan", response_model=ScanOut)
async def scan(file: UploadFile = File(...), preferred_language: Optional[str] = None):
    filename = file.filename or "upload"
    content_type = file.content_type or ""
    b = await file.read()
    enforce_file_size(b)

    t = infer_type(filename, content_type)
    if t == "unknown":
        raise HTTPException(status_code=415, detail="Unsupported file type. Use PDF, DOCX, TXT, PNG/JPG/JPEG.")

    extracted = ""
    input_type = t

    try:
        if t == "pdf":
            extracted, mode = extract_pdf_text(b)
            input_type = mode
        elif t == "docx":
            extracted = extract_docx_text(b)
        elif t == "txt":
            extracted = extract_txt_text(b)
        elif t == "image":
            extracted = extract_image_text(b)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file type.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    if not extracted or len(extracted) < 15:
        raise HTTPException(status_code=422, detail="No readable text found. For scanned docs, ensure OCR is enabled and scan is clear.")

    detected = detect_lang(extracted)
    preferred = resolve_preferred_lang(detected, preferred_language)

    scored = score_text(extracted)

    return {
        "filename": filename,
        "input_type": input_type,
        "detected_language": detected,
        "preferred_language": preferred,
        "extracted_text": extracted,
        "analysis": scored["executive"],
        "charts": scored["charts"],
        "checklist": scored["checklist"],
        "signals": scored["signals"],
        "meta": {
            **scored["meta"],
            "preferred_language": preferred,
            "disclaimer": "Not legal advice. Validate via official documents and due diligence.",
        },
    }
