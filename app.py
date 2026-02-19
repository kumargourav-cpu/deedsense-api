import os
import io
import re
import json
import shutil
import tempfile
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

# --- Text extraction deps ---
from PIL import Image
import pytesseract

from pypdf import PdfReader
from pdf2image import convert_from_bytes
from docx import Document

# --- OpenAI (stable call path) ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # Allows app to boot even if openai missing

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "DeedSense API")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))  # hard limit
MAX_PDF_PAGES_OCR = int(os.getenv("MAX_PDF_PAGES_OCR", "6"))  # scanned pdf OCR pages limit
MAX_CHARS_TO_MODEL = int(os.getenv("MAX_CHARS_TO_MODEL", "12000"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change if you want
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---- helpers: CORS parsing ----
def parse_origins(v: str) -> List[str]:
    v = (v or "").strip()
    if not v or v == "*":
        return ["*"]
    parts = [p.strip() for p in v.split(",") if p.strip()]
    return parts or ["*"]


app = FastAPI(title=APP_NAME, version="1.0.0")

origins = parse_origins(ALLOWED_ORIGINS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request models ----
class AnalyzeTextRequest(BaseModel):
    text: str
    source: Optional[str] = "text"


# ---- Safe limits ----
def _ensure_size_limit(file: UploadFile):
    # We can't know exact size without reading; enforce while reading
    pass


def clamp_text(text: str, max_chars: int = MAX_CHARS_TO_MODEL) -> str:
    text = (text or "").strip()
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def clean_for_analysis(text: str) -> str:
    """Remove extremely sensitive patterns (basic). You can expand this later."""
    t = text or ""
    # Mask emails + phone-ish patterns lightly (not perfect)
    t = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[email_redacted]", t)
    t = re.sub(r"\+?\d[\d\-\s]{7,}\d", "[phone_redacted]", t)
    return t


# ---- OCR availability checks ----
def is_tesseract_available() -> bool:
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def is_poppler_available() -> bool:
    return shutil.which("pdftoppm") is not None or shutil.which("pdftocairo") is not None


# ---- Extraction: DOCX ----
def extract_text_from_docx(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        doc = Document(tmp.name)
        parts = []
        for p in doc.paragraphs:
            if p.text and p.text.strip():
                parts.append(p.text.strip())
        return "\n".join(parts).strip()


# ---- Extraction: Image OCR ----
def ocr_image_bytes(data: bytes) -> str:
    if not is_tesseract_available():
        raise HTTPException(status_code=500, detail="OCR not available (tesseract missing)")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        txt = pytesseract.image_to_string(img)
        return (txt or "").strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image OCR failed: {str(e)}")


# ---- Extraction: PDF text ----
def extract_text_from_pdf_text_layer(data: bytes) -> str:
    """Extract selectable text (no OCR)."""
    try:
        reader = PdfReader(io.BytesIO(data))
        out = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                out.append(t)
        return "\n".join(out).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF text extraction failed: {str(e)}")


def ocr_scanned_pdf(data: bytes, max_pages: int = MAX_PDF_PAGES_OCR) -> str:
    """Convert first N pages to images then OCR each page."""
    if not is_poppler_available():
        raise HTTPException(status_code=500, detail="PDF OCR not available (poppler missing)")
    if not is_tesseract_available():
        raise HTTPException(status_code=500, detail="PDF OCR not available (tesseract missing)")

    try:
        # dpi 200 = good balance
        images = convert_from_bytes(data, dpi=200, first_page=1, last_page=max_pages)
        chunks = []
        for idx, img in enumerate(images, start=1):
            page_txt = pytesseract.image_to_string(img)
            page_txt = (page_txt or "").strip()
            if page_txt:
                chunks.append(f"[PAGE {idx}]\n{page_txt}")
        return "\n\n".join(chunks).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scanned PDF OCR failed: {str(e)}")


def smart_extract_pdf(data: bytes) -> Dict[str, Any]:
    """
    Strategy:
    1) Try text-layer extraction.
    2) If text is too small, OCR first N pages.
    """
    text_layer = extract_text_from_pdf_text_layer(data)
    if len(text_layer.strip()) >= 200:
        return {"method": "pdf_text_layer", "text": text_layer}

    # fallback to OCR if looks scanned
    ocr_text = ocr_scanned_pdf(data)
    return {"method": "pdf_ocr", "text": ocr_text}


# ---- AI Scan (OpenAI) ----
def local_heuristic_scan(text: str) -> Dict[str, Any]:
    """Fallback scoring when OpenAI is unavailable or quota exceeded."""
    t = text.lower()

    patterns = {
        "urgency_pressure": [
            "today only", "last chance", "limited time", "only today", "book now",
            "final offer", "ending soon", "before it’s gone", "few units left"
        ],
        "price_anchoring": ["was ", "before ", "discount", "save ", "reduced", "special price"],
        "vague_terms": ["guaranteed", "assured", "no risk", "100%", "sure shot", "best deal"],
        "missing_details": ["dm for", "call for", "details later", "will share", "on request"],
        "payment_risk": ["cash only", "no receipt", "off the record", "under table", "outside contract"],
    }

    scores = {}
    hits = {}
    for k, lst in patterns.items():
        count = sum(1 for p in lst if p in t)
        scores[k] = min(100, count * 25)
        if count:
            hits[k] = [p for p in lst if p in t][:5]

    overall = int(min(100, sum(scores.values()) / max(1, len(scores))))
    confidence = 0.55 if overall > 0 else 0.45

    risks = []
    if scores.get("urgency_pressure", 0) >= 25:
        risks.append("High-pressure urgency language detected (can signal manipulation).")
    if scores.get("payment_risk", 0) >= 25:
        risks.append("Potential payment/receipt risk language detected (verify receipts/contract terms).")
    if scores.get("missing_details", 0) >= 25:
        risks.append("Key details appear withheld ('call/DM for details'). Ask for written terms.")
    if scores.get("vague_terms", 0) >= 25:
        risks.append("Overconfident claims detected ('guaranteed', 'no risk'). Validate via documents.")

    summary = "Heuristic scan completed (AI unavailable). Review key risk signals and verify with official documents."

    return {
        "mode": "heuristic",
        "summary": summary,
        "overall_risk_score": overall,
        "confidence": confidence,
        "scores": scores,
        "signals": hits,
        "risks": risks[:8],
        "recommendations": [
            "Request full payment schedule and refund clauses in writing.",
            "Verify developer / title deed / escrow account details.",
            "Avoid cash/off-contract payments. Insist on receipts and official invoices."
        ],
    }


def openai_scan(text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY or not OpenAI:
        raise RuntimeError("OpenAI not configured")

    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "You are DeedSense, a trust & manipulation risk scanner for property investors. "
        "Analyze the provided text (listing, broker message, deed notes, payment terms, chat). "
        "Return ONLY valid JSON matching the requested schema. Do not include markdown."
    )

    schema_hint = {
        "summary": "string: actionable summary in 4-7 bullet-like sentences",
        "overall_risk_score": "integer 0-100",
        "confidence": "number 0-1",
        "scores": {
            "urgency_pressure": "0-100",
            "missing_details": "0-100",
            "payment_risk": "0-100",
            "legal_red_flags": "0-100",
            "too_good_to_be_true": "0-100",
        },
        "signals": [
            {"label": "string", "evidence": "short quote", "why_it_matters": "string"}
        ],
        "risks": ["string"],
        "recommendations": ["string"],
        "disclaimer": "string"
    }

    user = (
        "Analyze this property-related text. "
        "Focus on manipulation cues, due diligence gaps, payment/contract risks, and missing proof.\n\n"
        f"TEXT:\n{text}\n\n"
        "Return JSON in this schema (keys must match):\n"
        f"{json.dumps(schema_hint)}"
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        content = content.strip()

        # Ensure JSON only (strip common junk)
        content = re.sub(r"^\s*```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```\s*$", "", content).strip()

        data = json.loads(content)

        # minimal validation / defaults
        data.setdefault("disclaimer", "Not legal advice. Verify with official documents and professional due diligence.")
        data["mode"] = "openai"
        return data

    except Exception as e:
        raise RuntimeError(str(e))


def run_scan_pipeline(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text or ""
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted/provided to analyze.")

    trimmed = clamp_text(raw_text, MAX_CHARS_TO_MODEL)
    cleaned = clean_for_analysis(trimmed)

    # Try OpenAI first; fallback to heuristic
    try:
        result = openai_scan(cleaned)
        return result
    except Exception as e:
        # common quota error 429 etc
        fallback = local_heuristic_scan(cleaned)
        fallback["openai_error"] = str(e)
        return fallback


# ---- Routes ----
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": APP_NAME,
        "ocr_available": is_tesseract_available(),
        "poppler_available": is_poppler_available(),
        "openai_configured": bool(OPENAI_API_KEY),
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_pdf_pages_ocr": MAX_PDF_PAGES_OCR,
    }


@app.post("/analyze")
def analyze_text(payload: AnalyzeTextRequest):
    txt = payload.text or ""
    result = run_scan_pipeline(txt)
    return {
        "source": payload.source or "text",
        "extracted_text_preview": clamp_text(txt, 800),
        "result": result,
    }


@app.post("/analyze-file")
async def analyze_file(request: Request, file: UploadFile = File(...)):
    # Read safely with size cap
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    data = await file.read()
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB}MB.")

    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()

    extracted = ""
    method = "unknown"

    # Decide by extension first, then by content-type
    try:
        if filename.endswith(".pdf") or "pdf" in content_type:
            out = smart_extract_pdf(data)
            method = out["method"]
            extracted = out["text"]

        elif filename.endswith(".docx") or "word" in content_type or "officedocument" in content_type:
            extracted = extract_text_from_docx(data)
            method = "docx"

        elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or "image/" in content_type:
            extracted = ocr_image_bytes(data)
            method = "image_ocr"

        elif filename.endswith(".txt") or "text/plain" in content_type:
            extracted = data.decode("utf-8", errors="ignore")
            method = "txt"

        else:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Upload PDF, DOCX, TXT, JPG, PNG."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

    extracted = (extracted or "").strip()
    if not extracted:
        raise HTTPException(status_code=400, detail="No text could be extracted from this file.")

    result = run_scan_pipeline(extracted)

    return {
        "source": "file",
        "filename": file.filename,
        "content_type": file.content_type,
        "extraction_method": method,
        "extracted_text_preview": clamp_text(extracted, 1200),
        "result": result,
    }
