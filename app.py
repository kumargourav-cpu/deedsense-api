import os
import io
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import jwt  # PyJWT
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional DB (for usage/history)
import psycopg2
from psycopg2.extras import RealDictCursor

# Extraction libs
from pypdf import PdfReader
import docx  # python-docx
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes


# -----------------------------
# Config
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")  # optional but recommended
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
DEFAULT_FREE_SCANS = int(os.getenv("DEFAULT_FREE_SCANS", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))
OCR_LANG = os.getenv("OCR_LANG", "eng")  # e.g. "eng" or "eng+ara"


app = FastAPI(title="DeedSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DB Helpers (optional)
# -----------------------------
def db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def db_enabled() -> bool:
    return bool(DATABASE_URL)


def init_db():
    if not db_enabled():
        return
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            create table if not exists subscriptions (
              user_id text primary key,
              plan text not null default 'free',
              status text default 'active'
            );

            create table if not exists usage_monthly (
              user_id text not null,
              month text not null,
              scans_used int not null default 0,
              primary key (user_id, month)
            );

            create table if not exists scan_history (
              id uuid default gen_random_uuid() primary key,
              user_id text not null,
              created_at timestamptz not null default now(),
              input_type text not null,
              filename text,
              extracted_text_hash text,
              result_json jsonb not null
            );

            create index if not exists idx_scan_history_user on scan_history(user_id, created_at desc);
            """)
        conn.commit()


@app.on_event("startup")
def on_startup():
    init_db()


# -----------------------------
# Auth Helpers (Supabase JWT)
# -----------------------------
def get_user_from_bearer(auth_header: Optional[str]) -> Optional[Dict[str, Any]]:
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.replace("Bearer ", "").strip()
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="SUPABASE_JWT_SECRET not configured")

    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("role"),
        }
    except Exception:
        return None


def month_key_now() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def get_subscription(user_id: str) -> Dict[str, Any]:
    # If DB not enabled, default free and unlimited behavior handled outside
    if not db_enabled():
        return {"user_id": user_id, "plan": "free", "status": "active"}

    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("select * from subscriptions where user_id=%s", (user_id,))
            row = cur.fetchone()
            if not row:
                cur.execute("insert into subscriptions (user_id, plan, status) values (%s, 'free', 'active')", (user_id,))
                conn.commit()
                return {"user_id": user_id, "plan": "free", "status": "active"}
            return dict(row)


def get_usage(user_id: str) -> int:
    if not db_enabled():
        return 0
    mk = month_key_now()
    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("select scans_used from usage_monthly where user_id=%s and month=%s", (user_id, mk))
            row = cur.fetchone()
            return int(row["scans_used"]) if row else 0


def increment_usage(user_id: str) -> int:
    if not db_enabled():
        return 0
    mk = month_key_now()
    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                insert into usage_monthly (user_id, month, scans_used)
                values (%s, %s, 1)
                on conflict (user_id, month)
                do update set scans_used = usage_monthly.scans_used + 1
                returning scans_used
                """,
                (user_id, mk),
            )
            row = cur.fetchone()
            conn.commit()
            return int(row["scans_used"])


# -----------------------------
# Scan Logic (MVP placeholder)
# Replace with your AI later
# -----------------------------
def basic_risk_scan(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    signals: List[str] = []
    score = 0

    def hit(phrase: str, points: int, reason: str):
        nonlocal score
        if phrase in t:
            signals.append(reason)
            score += points

    hit("limited time", 15, "Urgency pressure detected")
    hit("last unit", 15, "Scarcity language detected")
    hit("guaranteed returns", 20, "Suspicious guarantee claim detected")
    hit("no questions asked", 10, "Over-reassurance pressure detected")
    hit("book now", 10, "CTA pressure detected")

    score = min(score, 100)
    label = "Low" if score < 20 else "Medium" if score < 50 else "High"

    return {
        "risk_score": score,
        "risk_label": label,
        "signals": signals,
        "summary": "MVP risk signal summary. Replace with your model output later.",
        "confidence": 0.72,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


# -----------------------------
# Safe text extraction
# -----------------------------
def enforce_file_size(file_bytes: bytes):
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB}MB.")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Strategy:
    1) Try text extraction with pypdf (works for normal PDFs)
    2) If extracted text is too small, assume scanned PDF and run OCR
    """
    enforce_file_size(file_bytes)

    # 1) Normal PDF text extraction
    extracted = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        extracted = "\n".join(parts).strip()
    except Exception:
        extracted = ""

    # 2) OCR fallback for scanned PDFs (or encrypted/unreadable)
    if len(extracted) < 40:
        images = convert_from_bytes(file_bytes, dpi=220)  # poppler required
        ocr_parts = []
        for img in images[:20]:  # safety: limit pages
            ocr_parts.append(pytesseract.image_to_string(img, lang=OCR_LANG))
        extracted = "\n".join(ocr_parts).strip()

    return extracted


def extract_text_from_docx(file_bytes: bytes) -> str:
    enforce_file_size(file_bytes)
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs]).strip()


def extract_text_from_image(file_bytes: bytes) -> str:
    enforce_file_size(file_bytes)
    img = Image.open(io.BytesIO(file_bytes))
    # convert to RGB for safety
    img = img.convert("RGB")
    return pytesseract.image_to_string(img, lang=OCR_LANG).strip()


def extract_text_from_txt(file_bytes: bytes) -> str:
    enforce_file_size(file_bytes)
    # Try utf-8 with fallback
    try:
        return file_bytes.decode("utf-8").strip()
    except Exception:
        return file_bytes.decode("latin-1", errors="ignore").strip()


# -----------------------------
# API Models
# -----------------------------
class AnalyzeTextIn(BaseModel):
    text: str


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "db": bool(DATABASE_URL),
        "ocr": True,  # if container built correctly
        "max_upload_mb": MAX_UPLOAD_MB,
        "ocr_lang": OCR_LANG,
    }


@app.get("/me")
def me(authorization: Optional[str] = Header(default=None)):
    u = get_user_from_bearer(authorization)
    if not u or not u.get("user_id"):
        return {"signed_in": False}

    sub = get_subscription(u["user_id"])
    usage = get_usage(u["user_id"]) if db_enabled() else None
    return {
        "signed_in": True,
        "user": u,
        "subscription": sub,
        "usage_month": usage,
        "free_limit": DEFAULT_FREE_SCANS,
        "db_enabled": db_enabled(),
    }


@app.post("/analyze-text")
def analyze_text(payload: AnalyzeTextIn, authorization: Optional[str] = Header(default=None)):
    u = get_user_from_bearer(authorization)
    if not u or not u.get("user_id"):
        raise HTTPException(status_code=401, detail="Sign in required.")

    user_id = u["user_id"]
    sub = get_subscription(user_id)
    plan = (sub.get("plan") or "free").lower()
    status = (sub.get("status") or "active").lower()

    # If DB not enabled, don't enforce usage
    if db_enabled():
        if plan != "free" and status not in ["active", "trialing"]:
            raise HTTPException(status_code=402, detail="Subscription not active.")

        if plan == "free":
            used = get_usage(user_id)
            if used >= DEFAULT_FREE_SCANS:
                raise HTTPException(status_code=402, detail=f"Free scans limit reached ({DEFAULT_FREE_SCANS}/month).")
            new_used = increment_usage(user_id)
        else:
            new_used = get_usage(user_id)
    else:
        new_used = None

    result = basic_risk_scan(payload.text)

    if db_enabled():
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "insert into scan_history (user_id, input_type, filename, extracted_text_hash, result_json) values (%s, %s, %s, %s, %s)",
                    (user_id, "text", None, sha256_text(payload.text), json.dumps(result)),
                )
            conn.commit()

    return {"plan": plan, "usage_month": new_used, "extracted_text": payload.text, "result": result}


@app.post("/extract-and-analyze")
async def extract_and_analyze(
    authorization: Optional[str] = Header(default=None),
    file: UploadFile = File(...),
):
    """
    Accepts: PDF/DOCX/TXT/PNG/JPG/JPEG
    Extracts text safely -> runs scan -> returns extracted text + result
    """
    u = get_user_from_bearer(authorization)
    if not u or not u.get("user_id"):
        raise HTTPException(status_code=401, detail="Sign in required.")

    user_id = u["user_id"]
    sub = get_subscription(user_id)
    plan = (sub.get("plan") or "free").lower()
    status = (sub.get("status") or "active").lower()

    # usage enforcement only if DB enabled
    if db_enabled():
        if plan != "free" and status not in ["active", "trialing"]:
            raise HTTPException(status_code=402, detail="Subscription not active.")

        if plan == "free":
            used = get_usage(user_id)
            if used >= DEFAULT_FREE_SCANS:
                raise HTTPException(status_code=402, detail=f"Free scans limit reached ({DEFAULT_FREE_SCANS}/month).")
            new_used = increment_usage(user_id)
        else:
            new_used = get_usage(user_id)
    else:
        new_used = None

    filename = file.filename or "upload"
    content = await file.read()
    enforce_file_size(content)

    # Determine type
    name = filename.lower()
    ctype = (file.content_type or "").lower()

    extracted_text = ""
    input_type = "file"

    try:
        if name.endswith(".pdf") or "pdf" in ctype:
            input_type = "pdf"
            extracted_text = extract_text_from_pdf(content)

        elif name.endswith(".docx") or "officedocument.wordprocessingml.document" in ctype:
            input_type = "docx"
            extracted_text = extract_text_from_docx(content)

        elif name.endswith(".txt") or "text/plain" in ctype:
            input_type = "txt"
            extracted_text = extract_text_from_txt(content)

        elif any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) or "image/" in ctype:
            input_type = "image"
            extracted_text = extract_text_from_image(content)

        else:
            raise HTTPException(status_code=415, detail="Unsupported file type. Use PDF, DOCX, TXT, PNG, JPG, JPEG.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

    if not extracted_text or len(extracted_text.strip()) < 10:
        raise HTTPException(status_code=422, detail="No readable text found. If this is a scanned file, ensure OCR is enabled and the scan is clear.")

    result = basic_risk_scan(extracted_text)

    if db_enabled():
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "insert into scan_history (user_id, input_type, filename, extracted_text_hash, result_json) values (%s, %s, %s, %s, %s)",
                    (user_id, input_type, filename, sha256_text(extracted_text), json.dumps(result)),
                )
            conn.commit()

    return {
        "plan": plan,
        "usage_month": new_used,
        "input_type": input_type,
        "filename": filename,
        "extracted_text": extracted_text,
        "result": result,
    }


@app.get("/history")
def history(authorization: Optional[str] = Header(default=None)):
    if not db_enabled():
        return {"items": [], "note": "DATABASE_URL not set. Enable Postgres to store scan history."}

    u = get_user_from_bearer(authorization)
    if not u or not u.get("user_id"):
        raise HTTPException(status_code=401, detail="Sign in required.")

    user_id = u["user_id"]
    sub = get_subscription(user_id)
    plan = (sub.get("plan") or "free").lower()

    if plan == "free":
        return {"items": [], "note": "History is available on Pro/Enterprise (later)."}

    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "select id, created_at, input_type, filename, result_json from scan_history where user_id=%s order by created_at desc limit 50",
                (user_id,),
            )
            return {"items": cur.fetchall()}
