import os
import io
import re
import json
import math
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Extraction libs
from pypdf import PdfReader
import docx  # python-docx
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes

# Optional: history in Postgres later
import psycopg2
from psycopg2.extras import RealDictCursor


# -----------------------------
# ENV CONFIG
# -----------------------------
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "18"))
OCR_LANG = os.getenv("OCR_LANG", "eng+ara")   # good for UAE context
PDF_OCR_MAX_PAGES = int(os.getenv("PDF_OCR_MAX_PAGES", "20"))
DATABASE_URL = os.getenv("DATABASE_URL")  # optional


# -----------------------------
# APP
# -----------------------------
app = FastAPI(title="DeedSense API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# DB (Optional)
# -----------------------------
def db_enabled() -> bool:
    return bool(DATABASE_URL)

def db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    if not db_enabled():
        return
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            create table if not exists scan_history_public (
              id uuid default gen_random_uuid() primary key,
              created_at timestamptz not null default now(),
              input_type text not null,
              filename text,
              lang text,
              extracted_text_hash text,
              result_json jsonb not null
            );
            create index if not exists idx_scan_history_public_created on scan_history_public(created_at desc);
            """)
        conn.commit()

@app.on_event("startup")
def on_startup():
    init_db()


# -----------------------------
# UTILS
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def enforce_file_size(file_bytes: bytes):
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB}MB.")

def sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

def safe_strip(s: str) -> str:
    return (s or "").replace("\x00", "").strip()

def clamp(n: float, a: float, b: float) -> float:
    return max(a, min(b, n))

def detect_language_rough(text: str) -> str:
    """
    Lightweight heuristic (no external services):
    - Arabic detection via Unicode ranges
    - Hindi via Devanagari range
    - else English
    """
    t = text or ""
    if re.search(r"[\u0600-\u06FF]", t):
        return "ar"
    if re.search(r"[\u0900-\u097F]", t):
        return "hi"
    # crude French/Spanish hints:
    if re.search(r"\b(le|la|les|des|une|un|et|avec)\b", t.lower()):
        return "fr"
    if re.search(r"\b(el|la|los|las|una|un|y|con)\b", t.lower()):
        return "es"
    return "en"


# -----------------------------
# SAFE TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Strategy:
    1) Try normal PDF text extraction (pypdf)
    2) If too little text, OCR scan pages via pdf2image + pytesseract
    """
    enforce_file_size(file_bytes)
    meta = {"mode": "pdf-text", "pages": None, "ocr_used": False}

    extracted = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        meta["pages"] = len(reader.pages)
        parts = []
        for page in reader.pages[:100]:
            parts.append(page.extract_text() or "")
        extracted = safe_strip("\n".join(parts))
    except Exception:
        extracted = ""

    # OCR fallback
    if len(extracted) < 60:
        meta["mode"] = "pdf-ocr"
        meta["ocr_used"] = True

        try:
            images = convert_from_bytes(file_bytes, dpi=220)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF OCR failed (poppler missing?): {str(e)}")

        ocr_parts = []
        for img in images[:PDF_OCR_MAX_PAGES]:
            ocr_parts.append(pytesseract.image_to_string(img, lang=OCR_LANG))
        extracted = safe_strip("\n".join(ocr_parts))

    return extracted, meta


def extract_text_from_docx(file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    enforce_file_size(file_bytes)
    d = docx.Document(io.BytesIO(file_bytes))
    txt = "\n".join([p.text for p in d.paragraphs])
    return safe_strip(txt), {"mode": "docx", "ocr_used": False}


def extract_text_from_txt(file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    enforce_file_size(file_bytes)
    try:
        return safe_strip(file_bytes.decode("utf-8")), {"mode": "txt", "ocr_used": False}
    except Exception:
        return safe_strip(file_bytes.decode("latin-1", errors="ignore")), {"mode": "txt", "ocr_used": False}


def extract_text_from_image(file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    enforce_file_size(file_bytes)
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    txt = pytesseract.image_to_string(img, lang=OCR_LANG)
    return safe_strip(txt), {"mode": "image-ocr", "ocr_used": True}


# -----------------------------
# ANALYSIS ENGINE (NO OPENAI)
# Investor-grade structured heuristics
# -----------------------------
RISK_DIMENSIONS = [
    ("manipulation", "Manipulation & Pressure"),
    ("claims", "Claims & Guarantees"),
    ("legal", "Legal & Compliance"),
    ("payments", "Payments & Financial Risk"),
    ("documentation", "Documentation Gaps"),
    ("transparency", "Transparency & Verifiability"),
    ("identity", "Identity & Authority Signals"),
]

PATTERNS = {
    "urgency": [
        r"\blimited time\b",
        r"\bonly today\b",
        r"\blast chance\b",
        r"\bfinal offer\b",
        r"\bbook now\b",
        r"\bact now\b",
        r"\b24 hours\b",
    ],
    "scarcity": [
        r"\blast unit\b",
        r"\bonly \d+ left\b",
        r"\bexclusive\b",
        r"\bnot available later\b",
    ],
    "guarantees": [
        r"\bguaranteed returns?\b",
        r"\b100% guarantee\b",
        r"\bno risk\b",
        r"\brisk[- ]free\b",
        r"\bprofit guaranteed\b",
    ],
    "anti_due_diligence": [
        r"\bno questions asked\b",
        r"\btrust me\b",
        r"\bdon't worry about\b",
        r"\bno need to\b.*\bverify\b",
    ],
    "document_gaps": [
        r"\bnot available\b.*\bdocuments?\b",
        r"\bwill share later\b",
        r"\bon request\b",
        r"\bverbally\b",
    ],
    "authority_push": [
        r"\bCEO\b",
        r"\bchairman\b",
        r"\bgovernment\b",
        r"\bofficial\b",
        r"\blicensed\b",
        r"\bapproved\b",
    ],
    "payment_risk": [
        r"\bcash only\b",
        r"\bcrypto\b",
        r"\buntraceable\b",
        r"\badvance\b.*\bnon[- ]refundable\b",
        r"\bdeposit\b.*\bno refund\b",
        r"\bpay to personal\b",
    ],
}

NEGATIVE_SIGNALS = [
    ("contains_escrow", r"\bescrow\b|\btrust account\b|\bRERA\b|\bDLD\b", -6),
    ("has_disclaimer", r"\bsubject to\b|\bterms apply\b|\bverify\b|\bdue diligence\b", -4),
    ("provides_docs", r"\btitle deed\b|\bOqood\b|\bSPA\b|\bMOU\b|\bpassport\b|\btrade license\b", -5),
]

def score_text(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    length = len(t)

    # Base dimensional scoring buckets
    dims = {k: 0.0 for (k, _) in RISK_DIMENSIONS}
    evidence = {k: [] for (k, _) in RISK_DIMENSIONS}

    def add(dim: str, points: float, note: str):
        dims[dim] += points
        evidence[dim].append(note)

    # Patterns → dimensions
    def find_hits(key: str) -> int:
        cnt = 0
        for p in PATTERNS[key]:
            if re.search(p, t):
                cnt += 1
        return cnt

    urg = find_hits("urgency")
    sca = find_hits("scarcity")
    gue = find_hits("guarantees")
    anti = find_hits("anti_due_diligence")
    docg = find_hits("document_gaps")
    auth = find_hits("authority_push")
    pay = find_hits("payment_risk")

    if urg:
        add("manipulation", 10 + urg * 6, f"Urgency language detected ({urg} signal(s))")
    if sca:
        add("manipulation", 8 + sca * 5, f"Scarcity / exclusivity language detected ({sca} signal(s))")
    if gue:
        add("claims", 14 + gue * 7, f"Guarantee-style claims detected ({gue} signal(s))")
        add("transparency", 6 + gue * 3, "High-precision claims without verifiable proofs often correlate with mis-selling")
    if anti:
        add("manipulation", 10 + anti * 6, "Anti-verification or over-reassurance phrasing detected")
        add("documentation", 8 + anti * 5, "Pressure that discourages due diligence is a strong risk marker")
    if docg:
        add("documentation", 12 + docg * 6, "Missing/withheld documentation signals")
        add("transparency", 8 + docg * 4, "Delayed docs reduce verifiability")
    if auth:
        add("identity", 6 + auth * 2, "Authority branding detected (verify via license/registry)")
    if pay:
        add("payments", 14 + pay * 6, "High-risk payment instruction pattern detected")
        add("legal", 6 + pay * 2, "Payment methods can increase compliance and recovery risk")

    # Negative signals reduce risk (good signals)
    for name, pattern, delta in NEGATIVE_SIGNALS:
        if re.search(pattern, t):
            # spread reductions across key dims
            dims["documentation"] += delta
            dims["transparency"] += delta
            dims["legal"] += delta / 2
            evidence["documentation"].append(f"Positive sign: references verifiable controls ({name})")
            evidence["transparency"].append(f"Positive sign: references verifiable controls ({name})")

    # Normalize by length (short texts should not get extreme confidence)
    length_factor = clamp(length / 1200.0, 0.35, 1.0)

    # Build total score
    raw_total = sum(max(0.0, v) for v in dims.values())
    raw_total *= length_factor

    # Convert to 0..100 with a saturating curve
    score = 100 * (1 - math.exp(-raw_total / 85.0))
    score = clamp(score, 0.0, 100.0)

    # Label
    if score < 20:
        label = "Low"
    elif score < 45:
        label = "Guarded"
    elif score < 70:
        label = "High"
    else:
        label = "Critical"

    # Confidence: based on length + number of matched patterns
    hit_count = urg + sca + gue + anti + docg + auth + pay
    conf = 0.35 + 0.35 * length_factor + 0.03 * min(hit_count, 8)
    conf = clamp(conf, 0.35, 0.90)

    # Make category breakdown (0..100)
    dim_scores = {}
    for k, _ in RISK_DIMENSIONS:
        # scale each dim using soft cap
        v = max(0.0, dims[k]) * length_factor
        dim_scores[k] = float(clamp(100 * (1 - math.exp(-v / 28.0)), 0, 100))

    # Key recommendations (actionable)
    tips = build_tips(dim_scores, t)

    # Executive summary
    summary = build_summary(score, label, dim_scores, hit_count)

    # “What to verify” checklist
    checklist = build_checklist(dim_scores)

    # Red flags (top evidence)
    top_flags = []
    for k, _ in RISK_DIMENSIONS:
        for ev in evidence[k][:3]:
            if "Positive sign" not in ev:
                top_flags.append({"dimension": k, "note": ev})
    top_flags = top_flags[:10]

    return {
        "risk_score": round(score, 1),
        "risk_label": label,
        "confidence": round(conf, 2),
        "hit_count": hit_count,
        "dimensions": dim_scores,
        "summary": summary,
        "top_flags": top_flags,
        "recommendations": tips,
        "verification_checklist": checklist,
        "created_at": now_iso(),
    }

def build_summary(score: float, label: str, dim_scores: Dict[str, float], hits: int) -> str:
    # Focus on top 2 dims
    top2 = sorted(dim_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    def pretty(k: str) -> str:
        for kk, name in RISK_DIMENSIONS:
            if kk == k:
                return name
        return k
    if hits == 0 and score < 18:
        return (
            "The text contains relatively few high-risk manipulation markers. "
            "This does NOT confirm legitimacy — it only means the language itself isn’t strongly pressuring or deceptive. "
            "Proceed with standard verification (developer, agent license, escrow/payment proof, contract terms)."
        )
    return (
        f"Overall risk is **{label}** (score {round(score,1)}/100). "
        f"Highest risk areas: **{pretty(top2[0][0])}** and **{pretty(top2[1][0])}**. "
        "This score reflects language-based signals (pressure patterns, unverifiable claims, and documentation/payment risk cues) "
        "and should be validated with official documents and due diligence."
    )

def build_tips(dim_scores: Dict[str, float], t: str) -> List[Dict[str, Any]]:
    tips: List[Dict[str, Any]] = []

    def add(priority: str, title: str, details: str):
        tips.append({"priority": priority, "title": title, "details": details})

    if dim_scores["payments"] >= 45:
        add(
            "High",
            "Confirm payment safety before transferring anything",
            "Insist on escrow/trust account routes where applicable, written invoices, and beneficiary matching the legal entity. "
            "Avoid personal accounts, cash-only requests, or vague payment instructions."
        )

    if dim_scores["documentation"] >= 45 or dim_scores["transparency"] >= 45:
        add(
            "High",
            "Ask for a document pack — not screenshots",
            "Request: title deed/Oqood, SPA/MOU, payment plan schedule, developer project number, agent RERA/DLD license (UAE), "
            "and any reservation forms. Cross-check names, dates, and amounts."
        )

    if dim_scores["claims"] >= 40:
        add(
            "Medium",
            "Treat guaranteed returns as marketing, not proof",
            "If returns are promised, ask for the basis: rental comps, occupancy assumptions, fees, service charges, and exit liquidity. "
            "Verify who is guaranteeing and whether it is contractually enforceable."
        )

    if dim_scores["manipulation"] >= 40:
        add(
            "Medium",
            "Slow down the decision cycle",
            "Pressure tactics correlate with mis-selling. Use a 24–48 hour cooling-off rule, compare alternatives, "
            "and keep communication in writing."
        )

    # always add baseline
    add(
        "Baseline",
        "Verify identity & authority",
        "Confirm the agent/broker identity via official license registries where available; validate the developer/project via official portals."
    )
    add(
        "Baseline",
        "Keep evidence",
        "Save PDFs, emails, WhatsApp messages, receipts, and call summaries. These matter if disputes arise."
    )

    return tips[:8]

def build_checklist(dim_scores: Dict[str, float]) -> List[Dict[str, Any]]:
    items: List[Tuple[str, str, str]] = [
        ("Documents", "Title deed / Oqood / SPA / MOU", "Ask for originals or official PDFs; verify signatures and entity names."),
        ("Payments", "Beneficiary verification", "Ensure payment goes to the correct legal entity; match invoice and trade license."),
        ("Project", "Project registration / permit numbers", "Cross-check the project exists and the unit details match."),
        ("Fees", "Service charges, admin fees, commissions", "Demand full fee breakdown in writing."),
        ("Timelines", "Handover date + delay clauses", "Check penalty clauses and what constitutes delay."),
        ("Refunds", "Cancellation and refund conditions", "Look for non-refundable language and exceptions."),
        ("KYC", "Agent / broker license", "Verify license validity and authorized scope."),
        ("Proof", "Marketing claims evidence", "Request supporting data: comps, contracts, and audited statements if offered."),
    ]

    # promote stronger emphasis based on dimensions
    out = []
    for cat, title, why in items:
        weight = 1
        if cat == "Payments" and dim_scores["payments"] > 40:
            weight = 3
        if cat == "Documents" and dim_scores["documentation"] > 40:
            weight = 3
        if cat == "Proof" and dim_scores["claims"] > 35:
            weight = 2
        out.append({"category": cat, "item": title, "why": why, "priority": weight})

    out.sort(key=lambda x: x["priority"], reverse=True)
    return out


# -----------------------------
# I18N (limited, offline)
# -----------------------------
I18N = {
    "en": {
        "label": {"Low": "Low", "Guarded": "Guarded", "High": "High", "Critical": "Critical"},
        "disclaimer": "DeedSense provides language-based risk signals, not legal advice. Always verify with official documents and professional due diligence.",
    },
    "ar": {
        "label": {"Low": "منخفض", "Guarded": "حذر", "High": "مرتفع", "Critical": "حرِج"},
        "disclaimer": "يوفّر DeedSense إشارات مخاطر مبنية على لغة النص وليس نصيحة قانونية. يجب التحقق من الوثائق الرسمية وإجراء العناية الواجبة.",
    },
    "hi": {
        "label": {"Low": "कम", "Guarded": "सावधानी", "High": "उच्च", "Critical": "गंभीर"},
        "disclaimer": "DeedSense केवल भाषा-आधारित जोखिम संकेत देता है, यह कानूनी सलाह नहीं है। दस्तावेज़ों और ड्यू डिलिजेंस से सत्यापन ज़रूरी है।",
    },
    "fr": {
        "label": {"Low": "Faible", "Guarded": "Prudent", "High": "Élevé", "Critical": "Critique"},
        "disclaimer": "DeedSense fournit des signaux de risque basés sur le langage, pas un avis juridique. Vérifiez toujours via des documents officiels et une due diligence.",
    },
    "es": {
        "label": {"Low": "Bajo", "Guarded": "Precaución", "High": "Alto", "Critical": "Crítico"},
        "disclaimer": "DeedSense ofrece señales de riesgo basadas en el lenguaje, no asesoría legal. Verifica con documentos oficiales y due diligence.",
    },
}

def localize_result(result: Dict[str, Any], out_lang: str) -> Dict[str, Any]:
    lang = out_lang if out_lang in I18N else "en"
    label_map = I18N[lang]["label"]
    label = result.get("risk_label", "Low")
    result["risk_label_local"] = label_map.get(label, label)
    result["disclaimer_local"] = I18N[lang]["disclaimer"]
    return result


# -----------------------------
# API MODELS
# -----------------------------
class AnalyzeTextIn(BaseModel):
    text: str
    out_lang: Optional[str] = "en"


class ChatIn(BaseModel):
    message: str
    context_text: Optional[str] = ""
    out_lang: Optional[str] = "en"


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "max_upload_mb": MAX_UPLOAD_MB,
        "ocr_lang": OCR_LANG,
        "pdf_ocr_max_pages": PDF_OCR_MAX_PAGES,
        "db_enabled": db_enabled(),
        "time": now_iso(),
    }


@app.post("/analyze-text")
def analyze_text(payload: AnalyzeTextIn):
    text = safe_strip(payload.text)
    if len(text) < 10:
        raise HTTPException(status_code=422, detail="Text too short to analyze.")
    lang_inferred = detect_language_rough(text)
    result = score_text(text)
    result["lang_detected"] = lang_inferred
    result["input_type"] = "text"
    result = localize_result(result, payload.out_lang or "en")

    # optionally store (public, no user)
    if db_enabled():
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "insert into scan_history_public (input_type, filename, lang, extracted_text_hash, result_json) values (%s,%s,%s,%s,%s)",
                    ("text", None, lang_inferred, sha256_text(text), json.dumps(result)),
                )
            conn.commit()

    return {"extracted_text": text, "result": result}


@app.post("/extract-and-analyze")
async def extract_and_analyze(
    out_lang: str = Form("en"),
    file: UploadFile = File(...),
):
    filename = file.filename or "upload"
    content_type = (file.content_type or "").lower()
    raw = await file.read()
    enforce_file_size(raw)

    name = filename.lower()
    extracted = ""
    meta: Dict[str, Any] = {"filename": filename, "content_type": content_type}

    try:
        if name.endswith(".pdf") or "pdf" in content_type:
            extracted, m = extract_text_from_pdf(raw)
            meta.update(m)
            input_type = "pdf"

        elif name.endswith(".docx") or "officedocument.wordprocessingml.document" in content_type:
            extracted, m = extract_text_from_docx(raw)
            meta.update(m)
            input_type = "docx"

        elif name.endswith(".txt") or "text/plain" in content_type:
            extracted, m = extract_text_from_txt(raw)
            meta.update(m)
            input_type = "txt"

        elif any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]) or "image/" in content_type:
            extracted, m = extract_text_from_image(raw)
            meta.update(m)
            input_type = "image"

        else:
            raise HTTPException(status_code=415, detail="Unsupported file type. Use PDF, DOCX, TXT, PNG, JPG, JPEG.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

    extracted = safe_strip(extracted)
    if len(extracted) < 10:
        raise HTTPException(
            status_code=422,
            detail="No readable text found. If this is scanned, ensure OCR is enabled and the scan is clear."
        )

    lang_inferred = detect_language_rough(extracted)
    result = score_text(extracted)
    result["lang_detected"] = lang_inferred
    result["input_type"] = input_type
    result["extract_meta"] = meta
    result = localize_result(result, out_lang or "en")

    if db_enabled():
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "insert into scan_history_public (input_type, filename, lang, extracted_text_hash, result_json) values (%s,%s,%s,%s,%s)",
                    (input_type, filename, lang_inferred, sha256_text(extracted), json.dumps(result)),
                )
            conn.commit()

    return {"filename": filename, "extracted_text": extracted, "result": result}


@app.get("/history")
def history(limit: int = 50):
    if not db_enabled():
        return {"items": [], "note": "DATABASE_URL not set. UI will use local browser history for now."}

    limit = max(1, min(200, int(limit)))
    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "select id, created_at, input_type, filename, lang, result_json from scan_history_public order by created_at desc limit %s",
                (limit,),
            )
            return {"items": cur.fetchall()}


@app.post("/chat")
def chat(payload: ChatIn):
    """
    MVP chat:
    - Uses offline guidance and your scan context.
    - Web-research mode can be added later (server-side search integrations).
    """
    msg = safe_strip(payload.message)
    if not msg:
        raise HTTPException(status_code=422, detail="Message required.")

    ctx = safe_strip(payload.context_text or "")
    lang = payload.out_lang if payload.out_lang in I18N else "en"

    # Simple intent hints
    want_properties = bool(re.search(r"\b(find|search|options|properties|projects|area|budget|villa|apartment)\b", msg.lower()))
    want_risk_help = bool(re.search(r"\b(risk|scam|safe|legit|verify|fraud|trust)\b", msg.lower()))

    base = []
    base.append("Here’s a practical investor-grade approach based on your message.")

    if ctx and want_risk_help:
        # quick add-on: recommend verification steps
        base.append("If you paste the full listing/broker message and payment terms, I can pinpoint pressure language and missing-doc signals.")
        base.append("Immediate due diligence checklist: verify agent license, project registration, beneficiary verification, escrow/trust account, and contract clauses.")

    if want_properties:
        # no web search right now (by design)
        base.append("Property search mode is currently offline in this MVP build (no web queries yet).")
        base.append("If you share: city/area, budget, purpose (end-use vs investment), timeline, and risk tolerance — I’ll propose a shortlist framework and what to compare.")
        base.append("Later we can enable Web Research mode to return verified listings with links.")

    base.append("If you want, paste your document text and ask: “What are the top 5 red flags and what should I ask the agent?”")

    answer = "\n\n".join(base)

    # lightweight localization for disclaimers only
    disclaimer = I18N[lang]["disclaimer"]

    return {"reply": answer, "disclaimer": disclaimer, "mode": "offline-mvp"}
