import os
import io
import json
import time
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import jwt  # PyJWT
from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

# Optional DB (for usage/history/profile)
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
DATABASE_URL = os.getenv("DATABASE_URL")  # REQUIRED for profile enforcement
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://deedsense-ui.onrender.com")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

DEFAULT_FREE_SCANS = int(os.getenv("DEFAULT_FREE_SCANS", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))
OCR_LANG = os.getenv("OCR_LANG", "eng")  # e.g. "eng" or "eng+ara"

# Security / rate-limit
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))  # per IP
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))  # short burst


app = FastAPI(title="DeedSense API", version="1.1.0")

# CORS NOTE:
# If allow_credentials=True, do NOT use allow_origins=["*"] in production.
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Basic in-memory rate limiter (simple but effective on single Render instance)
# -----------------------------
_ip_hits: Dict[str, List[float]] = {}  # { ip: [timestamps...] }


def _cleanup_hits(ip: str, window_seconds: int = 60):
    now = time.time()
    hits = _ip_hits.get(ip, [])
    hits = [t for t in hits if now - t < window_seconds]
    _ip_hits[ip] = hits


def _rate_limit(ip: str):
    _cleanup_hits(ip)
    hits = _ip_hits.get(ip, [])
    # Allow small burst but limit sustained
    if len(hits) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    hits.append(time.time())
    _ip_hits[ip] = hits


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    # Rate-limit only sensitive endpoints
    if request.url.path in ["/extract", "/extract-and-analyze", "/analyze-text", "/history", "/profile", "/me"]:
        _rate_limit(ip)
    return await call_next(request)


# -----------------------------
# DB Helpers
# -----------------------------
def db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def db_enabled() -> bool:
    return bool(DATABASE_URL)


def init_db():
    if not db_enabled():
        # We want profile enforcement; no DB = no production readiness.
        return

    with db() as conn:
        with conn.cursor() as cur:
            # Needed for gen_random_uuid()
            cur.execute("create extension if not exists pgcrypto;")

            cur.execute("""
            create table if not exists profiles (
              user_id text primary key,
              email text,
              full_name text,
              country text,
              investor_type text,   -- e.g. 'End Buyer' | 'Investor' | 'Agent' | 'Developer'
              created_at timestamptz not null default now(),
              updated_at timestamptz not null default now()
            );

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


def require_user(authorization: Optional[str]) -> Dict[str, Any]:
    u = get_user_from_bearer(authorization)
    if not u or not u.get("user_id"):
        raise HTTPException(status_code=401, detail="Sign in required.")
    return u


def month_key_now() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def get_subscription(user_id: str) -> Dict[str, Any]:
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
# Profile enforcement
# -----------------------------
def get_profile(user_id: str) -> Optional[Dict[str, Any]]:
    if not db_enabled():
        return None
    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("select * from profiles where user_id=%s", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def require_profile(user_id: str):
    if not db_enabled():
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured (profile required).")
    prof = get_profile(user_id)
    # Minimal “complete profile” rule:
    if not prof or not prof.get("full_name") or not prof.get("country") or not prof.get("investor_type"):
        raise HTTPException(status_code=403, detail="PROFILE_REQUIRED")
    return prof


class ProfileIn(BaseModel):
    full_name: str
    country: str
    investor_type: str


# -----------------------------
# Scan Logic (MVP placeholder)
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
        "scores": {
            "risk": score,
            "trust": max(0, 100 - score),
            "manipulation": min(100, score + (10 if signals else 0)),
        },
        "risk_label": label,
        "red_flags": signals,
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


def extract_text_from_pdf(file_bytes: bytes) -> Dict[str, Any]:
    enforce_file_size(file_bytes)

    extracted = ""
    pages = 0
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = len(reader.pages)
        parts = [(page.extract_text() or "") for page in reader.pages]
        extracted = "\n".join(parts).strip()
    except Exception:
        extracted = ""

    meta = {"pages": pages, "ocr": False}

    # OCR fallback for scanned PDFs
    if len(extracted) < 40:
        images = convert_from_bytes(file_bytes, dpi=220)  # poppler required
        ocr_parts = []
        for img in images[:20]:  # safety: limit pages
            ocr_parts.append(pytesseract.image_to_string(img, lang=OCR_LANG))
        extracted = "\n".join(ocr_parts).strip()
        meta["ocr"] = True
        meta["pages"] = min(len(images), 20)

    return {"text": extracted, "meta": meta}


def extract_text_from_docx(file_bytes: bytes) -> Dict[str, Any]:
    enforce_file_size(file_bytes)
    doc = docx.Document(io.BytesIO(file_bytes))
    return {"text": "\n".join([p.text for p in doc.paragraphs]).strip(), "meta": {"ocr": False}}


def extract_text_from_image(file_bytes: bytes) -> Dict[str, Any]:
    enforce_file_size(file_bytes)
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return {"text": pytesseract.image_to_string(img, lang=OCR_LANG).strip(), "meta": {"ocr": True}}


def extract_text_from_txt(file_bytes: bytes) -> Dict[str, Any]:
    enforce_file_size(file_bytes)
    try:
        t = file_bytes.decode("utf-8").strip()
    except Exception:
        t = file_bytes.decode("latin-1", errors="ignore").strip()
    return {"text": t, "meta": {"ocr": False}}


# -----------------------------
# API Models
# -----------------------------
class AnalyzeTextIn(BaseModel):
    text: str
    preferred_language: Optional[str] = None


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "db": bool(DATABASE_URL),
        "ocr": True,
        "max_upload_mb": MAX_UPLOAD_MB,
        "ocr_lang": OCR_LANG,
        "allowed_origins": origins,
    }


@app.get("/me")
def me(authorization: Optional[str] = Header(default=None)):
    u = get_user_from_bearer(authorization)
    if not u or not u.get("user_id"):
        return {"signed_in": False}

    sub = get_subscription(u["user_id"])
    usage = get_usage(u["user_id"]) if db_enabled() else None
    prof = get_profile(u["user_id"]) if db_enabled() else None

    complete = bool(prof and prof.get("full_name") and prof.get("country") and prof.get("investor_type"))

    return {
        "signed_in": True,
        "user": u,
        "profile": prof,
        "profile_complete": complete,
        "subscription": sub,
        "usage_month": usage,
        "free_limit": DEFAULT_FREE_SCANS,
        "db_enabled": db_enabled(),
    }


@app.get("/profile")
def profile_get(authorization: Optional[str] = Header(default=None)):
    u = require_user(authorization)
    if not db_enabled():
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    prof = get_profile(u["user_id"])
    return {"profile": prof}


@app.post("/profile")
def profile_upsert(payload: ProfileIn, authorization: Optional[str] = Header(default=None)):
    u = require_user(authorization)
    if not db_enabled():
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")

    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into profiles (user_id, email, full_name, country, investor_type, updated_at)
                values (%s, %s, %s, %s, %s, now())
                on conflict (user_id)
                do update set full_name=excluded.full_name, country=excluded.country, investor_type=excluded.investor_type, updated_at=now()
                """,
                (u["user_id"], u.get("email"), payload.full_name, payload.country, payload.investor_type),
            )
        conn.commit()

    return {"ok": True}


def enforce_plan_and_usage(user_id: str) -> Dict[str, Any]:
    sub = get_subscription(user_id)
    plan = (sub.get("plan") or "free").lower()
    status = (sub.get("status") or "active").lower()

    # usage enforcement only if DB enabled
    usage_month = None
    if db_enabled():
        if plan != "free" and status not in ["active", "trialing"]:
            raise HTTPException(status_code=402, detail="Subscription not active.")

        if plan == "free":
            used = get_usage(user_id)
            if used >= DEFAULT_FREE_SCANS:
                raise HTTPException(status_code=402, detail=f"Free scans limit reached ({DEFAULT_FREE_SCANS}/month).")
            usage_month = increment_usage(user_id)
        else:
            usage_month = get_usage(user_id)

    return {"plan": plan, "usage_month": usage_month}


@app.post("/analyze-text")
def analyze_text(payload: AnalyzeTextIn, authorization: Optional[str] = Header(default=None)):
    u = require_user(authorization)

    # REQUIRE profile first (prevents anonymous abuse)
    require_profile(u["user_id"])

    usage = enforce_plan_and_usage(u["user_id"])
    result = basic_risk_scan(payload.text)

    if db_enabled():
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "insert into scan_history (user_id, input_type, filename, extracted_text_hash, result_json) values (%s, %s, %s, %s, %s)",
                    (u["user_id"], "text", None, sha256_text(payload.text), json.dumps(result)),
                )
            conn.commit()

    return {
        "title": "DeedSense Report",
        **usage,
        "extracted_text": payload.text,
        **result,
    }


@app.post("/extract")
async def extract_only(
    authorization: Optional[str] = Header(default=None),
    file: UploadFile = File(...),
):
    """
    UI expects: POST /extract -> { text, meta }
    """
    u = require_user(authorization)
    require_profile(u["user_id"])

    filename = file.filename or "upload"
    content = await file.read()
    enforce_file_size(content)

    name = filename.lower()
    ctype = (file.content_type or "").lower()

    try:
        if name.endswith(".pdf") or "pdf" in ctype:
            out = extract_text_from_pdf(content)
            out["meta"]["source"] = filename
            return out

        if name.endswith(".docx") or "officedocument.wordprocessingml.document" in ctype:
            out = extract_text_from_docx(content)
            out["meta"]["source"] = filename
            return out

        if name.endswith(".txt") or "text/plain" in ctype:
            out = extract_text_from_txt(content)
            out["meta"]["source"] = filename
            return out

        if any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) or "image/" in ctype:
            out = extract_text_from_image(content)
            out["meta"]["source"] = filename
            return out

        raise HTTPException(status_code=415, detail="Unsupported file type. Use PDF, DOCX, TXT, PNG, JPG, JPEG.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")


@app.post("/extract-and-analyze")
async def extract_and_analyze(
    authorization: Optional[str] = Header(default=None),
    file: UploadFile = File(...),
):
    """
    Accepts: PDF/DOCX/TXT/PNG/JPG/JPEG
    Extracts text safely -> runs scan -> returns extracted text + result
    """
    u = require_user(authorization)
    require_profile(u["user_id"])

    usage = enforce_plan_and_usage(u["user_id"])

    filename = file.filename or "upload"
    content = await file.read()
    enforce_file_size(content)

    name = filename.lower()
    ctype = (file.content_type or "").lower()

    extracted_text = ""
    input_type = "file"
    meta: Dict[str, Any] = {"source": filename}

    try:
        if name.endswith(".pdf") or "pdf" in ctype:
            input_type = "pdf"
            out = extract_text_from_pdf(content)
            extracted_text = out["text"]
            meta.update(out["meta"])

        elif name.endswith(".docx") or "officedocument.wordprocessingml.document" in ctype:
            input_type = "docx"
            out = extract_text_from_docx(content)
            extracted_text = out["text"]
            meta.update(out["meta"])

        elif name.endswith(".txt") or "text/plain" in ctype:
            input_type = "txt"
            out = extract_text_from_txt(content)
            extracted_text = out["text"]
            meta.update(out["meta"])

        elif any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) or "image/" in ctype:
            input_type = "image"
            out = extract_text_from_image(content)
            extracted_text = out["text"]
            meta.update(out["meta"])

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
                    (u["user_id"], input_type, filename, sha256_text(extracted_text), json.dumps(result)),
                )
            conn.commit()

    return {
        "title": "DeedSense Report",
        **usage,
        "input_type": input_type,
        "filename": filename,
        "meta": meta,
        "extracted_text": extracted_text,
        **result,
    }


@app.get("/history")
def history(authorization: Optional[str] = Header(default=None)):
    if not db_enabled():
        return {"items": [], "note": "DATABASE_URL not set. Enable Postgres to store scan history."}

    u = require_user(authorization)
    require_profile(u["user_id"])

    sub = get_subscription(u["user_id"])
    plan = (sub.get("plan") or "free").lower()

    if plan == "free":
        return {"items": [], "note": "History is available on Pro/Enterprise (later)."}

    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "select id, created_at, input_type, filename, result_json from scan_history where user_id=%s order by created_at desc limit 50",
                (u["user_id"],),
            )
            return {"items": cur.fetchall()}
