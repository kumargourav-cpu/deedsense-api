import os
import io
import re
import json
import hashlib
import tempfile
from typing import Any, Dict, Optional, Tuple, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Text extraction libs ---
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from PIL import Image

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None  # type: ignore
    TESSERACT_AVAILABLE = False

# --- OpenAI (optional; keep stub if you want) ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================
# Config
# =========================
API_NAME = "DeedSense API"
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

# Limits (safe defaults)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))  # 12MB default
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "80000"))  # prevent huge payloads / cost

ALLOWED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/jpeg",
    "image/png",
}
ALLOWED_EXT = {".pdf", ".docx", ".jpg", ".jpeg", ".png"}


# =========================
# App
# =========================
app = FastAPI(title=API_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS] if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Schemas
# =========================
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


# =========================
# Safety helpers
# =========================
def _safe_filename(name: str) -> str:
    name = name or "upload"
    # strip path components
    name = name.split("/")[-1].split("\\")[-1]
    # keep safe chars
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:120]


def _get_ext(name: str) -> str:
    name = (name or "").lower()
    for ext in [".pdf", ".docx", ".jpeg", ".jpg", ".png"]:
        if name.endswith(ext):
            return ext
    return ""


def _enforce_size_limit(raw: bytes) -> None:
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed is {MAX_UPLOAD_MB}MB.",
        )


def _normalize_text(text: str) -> str:
    # remove null bytes & weird control chars (keep tabs/newlines)
    text = text.replace("\x00", " ")
    text = re.sub(r"[^\S\r\n]+", " ", text)  # collapse whitespace (not newlines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # limit excessive newlines
    text = text.strip()

    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n\n[TRUNCATED]"
    return text


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# =========================
# Extractors
# =========================
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Uses pdfminer.six (pure python).
    """
    _enforce_size_limit(pdf_bytes)
    # Write to temp because pdfminer expects a path for best reliability
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        text = pdf_extract_text(tmp.name) or ""
    return _normalize_text(text)


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    """
    Uses python-docx (pure python).
    """
    _enforce_size_limit(docx_bytes)
    f = io.BytesIO(docx_bytes)
    try:
        doc = Document(f)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid DOCX file.")
    parts: List[str] = []
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())
    # tables too (optional but useful)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text)
            if row_text.strip():
                parts.append(row_text.strip())
    return _normalize_text("\n".join(parts))


def extract_text_from_image_bytes(img_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    OCR via pytesseract + Pillow.
    Requires Tesseract installed on the server.
    """
    _enforce_size_limit(img_bytes)

    if not TESSERACT_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="OCR not available on server (pytesseract/tesseract not installed).",
        )

    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Simple safety: downscale huge images to avoid memory/cpu spikes
    max_side = 2400
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / float(max(w, h))
        img = img.resize((int(w * ratio), int(h * ratio)))

    # OCR
    text = pytesseract.image_to_string(img) or ""
    meta = {"ocr": True, "width": img.size[0], "height": img.size[1]}
    return _normalize_text(text), meta


def extract_text_from_upload(file: UploadFile, raw: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Single entry point. Validates type and extracts.
    """
    filename = _safe_filename(file.filename or "upload")
    ext = _get_ext(filename)

    if file.content_type not in ALLOWED_MIME and ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Upload PDF, DOCX, JPG, PNG only. Got: {file.content_type}",
        )

    base_meta = {
        "filename": filename,
        "content_type": file.content_type,
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
    }

    if ext == ".pdf" or file.content_type == "application/pdf":
        return extract_text_from_pdf_bytes(raw), {**base_meta, "source": "pdf"}
    if ext == ".docx" or file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx_bytes(raw), {**base_meta, "source": "docx"}
    if ext in {".jpg", ".jpeg", ".png"} or file.content_type in {"image/jpeg", "image/png"}:
        text, ocr_meta = extract_text_from_image_bytes(raw)
        return text, {**base_meta, "source": "image", **ocr_meta}

    raise HTTPException(status_code=415, detail="Unsupported file type.")


# =========================
# Analysis (stub + optional OpenAI)
# =========================
def analyze_stub(text: str) -> AnalyzeResponse:
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

    confidence = min(0.35 + (len(text) / 3000.0), 0.9)
    risk_score = min(0.15 + 0.25 * urgency_hits + 0.20 * money_hits + 0.10 * vague_hits, 0.95)

    return AnalyzeResponse(
        summary="Risk scan complete. Review signals and confirm via official documents and due diligence.",
        risk_score=float(risk_score),
        confidence=float(confidence),
        signals=signals,
        details="This is an MVP risk signal, not legal advice. Always verify contract terms and payment proof.",
        scores={
            "overall": float(risk_score),
            "urgency": float(min(urgency_hits / 3.0, 1.0)),
            "payment_pressure": float(min(money_hits / 3.0, 1.0)),
            "vagueness": float(min(vague_hits / 3.0, 1.0)),
        },
    )


def analyze_with_openai(text: str) -> AnalyzeResponse:
    if OpenAI is None or not OPENAI_API_KEY:
        return analyze_stub(text)

    client = OpenAI(api_key=OPENAI_API_KEY)
    system = (
        "You are DeedSense, a trust and manipulation risk scanner for property investors. "
        "Return STRICT JSON only:\n"
        "{summary: string, risk_score: number 0..1, confidence: number 0..1, "
        "signals: [{title: string, severity:'low'|'medium'|'high'}], details: string, "
        "scores: {overall:number, urgency:number, vagueness:number, payment_pressure:number}}\n"
        "Be conservative, no legal advice."
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""

    # parse json
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()

    try:
        obj = json.loads(s)
    except Exception:
        return analyze_stub(text)

    return AnalyzeResponse(
        summary=obj.get("summary") or "Analysis complete.",
        risk_score=float(obj.get("risk_score", 0.35)),
        confidence=float(obj.get("confidence", 0.6)),
        signals=obj.get("signals") or [],
        details=obj.get("details") or "",
        scores=obj.get("scores"),
    )


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True, "ocr_available": TESSERACT_AVAILABLE}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    """
    Upload a PDF/DOCX/JPG/PNG and get extracted text back.
    """
    raw = await file.read()
    text, meta = extract_text_from_upload(file, raw)

    if not text or len(text) < 5:
        raise HTTPException(
            status_code=422,
            detail="No readable text found. If this is a scanned PDF/image, OCR might be needed.",
        )

    return {"text": text, "meta": meta}


@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    # optional "mode" if you want to force stub vs openai
    mode: str = Form("auto"),
):
    """
    Upload a file, extract text, then analyze.
    """
    raw = await file.read()
    text, meta = extract_text_from_upload(file, raw)

    if not text or len(text) < 5:
        raise HTTPException(
            status_code=422,
            detail="No readable text found. If this is a scanned PDF/image, OCR might be needed.",
        )

    if mode == "stub":
        result = analyze_stub(text)
    else:
        result = analyze_with_openai(text)

    payload = result.model_dump()
    payload["extraction_meta"] = meta
    payload["extracted_text_preview"] = text[:1200]
    return payload
