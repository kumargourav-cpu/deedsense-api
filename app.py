import io
import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, Header, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openai import OpenAI

# =====================
# ENV
# =====================
DATABASE_URL = os.getenv("DATABASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # choose your model
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALG = "HS256"

UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

FREE_SCANS = int(os.getenv("FREE_SCANS", "5"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# =====================
# APP
# =====================
app = FastAPI(title="DeedSense AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS] if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# DB
# =====================
Base = declarative_base()

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False) if engine else None


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    allow_anonymized_improvement = Column(Boolean, default=False)  # opt-in
    scans = relationship("Scan", back_populates="user")


class Scan(Base):
    __tablename__ = "scans"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    client_id = Column(String(128), nullable=True, index=True)  # anonymous identifier
    category = Column(String(50), default="general")
    input_text = Column(Text, nullable=False)
    result_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="scans")


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    scan_id = Column(Integer, ForeignKey("scans.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    client_id = Column(String(128), nullable=True)
    rating = Column(Integer, default=0)  # -1, 0, +1
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    if engine:
        Base.metadata.create_all(bind=engine)


init_db()

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_db():
    if not SessionLocal:
        raise HTTPException(500, "DATABASE_URL not configured")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =====================
# AUTH
# =====================
def create_token(user_id: int) -> str:
    payload = {"sub": str(user_id), "exp": datetime.utcnow() + timedelta(days=30)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        user_id = int(payload.get("sub"))
    except (JWTError, ValueError, TypeError):
        return None
    return db.query(User).filter(User.id == user_id).first()


# =====================
# REDIS (Upstash REST)
# =====================
def redis_get(key: str) -> Optional[str]:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return None
    r = requests.get(
        f"{UPSTASH_REDIS_REST_URL}/get/{key}",
        headers={"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"},
        timeout=10,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    return data.get("result")


def redis_incr(key: str) -> int:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        # fallback: no redis means no usage tracking (not recommended)
        return 999999
    r = requests.post(
        f"{UPSTASH_REDIS_REST_URL}/incr/{key}",
        headers={"Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}"},
        timeout=10,
    )
    if r.status_code != 200:
        return 999999
    return int(r.json().get("result", 999999))


# =====================
# SCHEMAS
# =====================
class SignupIn(BaseModel):
    email: str
    password: str
    allow_anonymized_improvement: bool = False


class LoginIn(BaseModel):
    email: str
    password: str


class AnalyzeIn(BaseModel):
    content: str = Field(min_length=10)
    category: str = Field(default="general")
    region: str = Field(default="global")  # global / uae / uk / etc.


class FeedbackIn(BaseModel):
    scan_id: int
    rating: int = Field(ge=-1, le=1)
    note: Optional[str] = None


# =====================
# HEALTH
# =====================
@app.get("/health")
def health():
    return {"ok": True}


# =====================
# AUTH ENDPOINTS
# =====================
@app.post("/auth/signup")
def signup(body: SignupIn, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already registered")
    ph = pwd_ctx.hash(body.password)
    u = User(email=email, password_hash=ph, allow_anonymized_improvement=body.allow_anonymized_improvement)
    db.add(u)
    db.commit()
    db.refresh(u)
    return {"token": create_token(u.id)}


@app.post("/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    u = db.query(User).filter(User.email == email).first()
    if not u or not pwd_ctx.verify(body.password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"token": create_token(u.id)}


@app.get("/me")
def me(user: Optional[User] = Depends(get_current_user)):
    if not user:
        return {"authenticated": False}
    return {
        "authenticated": True,
        "email": user.email,
        "allow_anonymized_improvement": user.allow_anonymized_improvement,
    }


# =====================
# PERSONALIZATION MEMORY
# =====================
def build_user_memory(db: Session, user: Optional[User], client_id: Optional[str]) -> str:
    """
    We 'remember' by summarizing user's last few scans and feedback.
    This is personalization, not training.
    """
    q = db.query(Scan).order_by(Scan.created_at.desc())
    if user:
        q = q.filter(Scan.user_id == user.id)
    elif client_id:
        q = q.filter(Scan.client_id == client_id)
    else:
        return ""

    recent = q.limit(5).all()
    if not recent:
        return ""

    # only include short summaries, never full raw text
    items = []
    for s in recent:
        try:
            rj = json.loads(s.result_json)
            items.append({
                "ptr_score": rj.get("ptr_score"),
                "risk_level": rj.get("risk_level"),
                "top_flags": (rj.get("red_flags_detected") or [])[:3],
                "date": s.created_at.isoformat()
            })
        except Exception:
            continue

    if not items:
        return ""

    return "User context (recent patterns, for personalization): " + json.dumps(items)


# =====================
# ANALYSIS ENGINE
# =====================
def heuristic_risk_breakdown(text: str) -> Dict[str, int]:
    t = text.lower()
    identity = 0
    financial = 0
    contract = 0
    manipulation = 0
    roi = 0

    # Manipulation signals
    for kw in ["today only", "urgent", "last chance", "don’t tell", "secret", "limited time", "act now"]:
        if kw in t:
            manipulation += 12

    # Financial red flags
    for kw in ["crypto", "usdt", "personal account", "send to my account", "transfer now", "wire today", "advance fee"]:
        if kw in t:
            financial += 14

    # Identity / authority red flags
    for kw in ["agent license", "registration number", "rera", "dld", "broker card", "company registration", "official email"]:
        # if mentioned, reduces identity risk slightly
        if kw in t:
            identity -= 4
    for kw in ["whatsapp only", "no email", "don’t call", "private number"]:
        if kw in t:
            identity += 10

    # Contract signals
    for kw in ["no contract", "no spa", "no receipt", "no escrow", "no invoice"]:
        if kw in t:
            contract += 12

    # ROI promise signals
    for kw in ["guaranteed return", "guaranteed roi", "20% monthly", "double your money", "risk free"]:
        if kw in t:
            roi += 16

    def clamp(x): return max(0, min(25, x))
    return {
        "identity_risk": clamp(identity),
        "financial_risk": clamp(financial),
        "contract_risk": clamp(contract),
        "manipulation_risk": clamp(manipulation),
        "roi_risk": clamp(roi),
    }


def calc_ptr_score(b: Dict[str, int]) -> int:
    # weighted sum -> 0..100
    raw = (b["identity_risk"]*1.1 + b["financial_risk"]*1.3 + b["contract_risk"]*1.0 + b["manipulation_risk"]*1.0 + b["roi_risk"]*1.2)
    score = int(min(100, max(0, round(raw * 1.6))))
    return score


def risk_level(score: int) -> str:
    if score >= 81:
        return "Severe Risk"
    if score >= 61:
        return "High Caution"
    if score >= 41:
        return "Elevated Review"
    if score >= 21:
        return "Needs Documentation"
    return "Low Risk"


def openai_generate_structured(content: str, category: str, region: str, memory: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY missing")

    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "You are DeedSense AI, a property due diligence risk assistant. "
        "Return concise, structured JSON fields only. "
        "Do not include markdown. Do not include extra keys."
    )

    # region hints
    region_note = ""
    if region.lower() == "uae":
        region_note = "UAE hints: escrow account checks, broker license/RERA, developer registration, DLD/RERA references."
    elif region.lower() == "global":
        region_note = "Global hints: verify seller identity, escrow usage, contract clarity, payment channel safety, unrealistic ROI."

    user = (
        f"{memory}\n\n"
        f"Category: {category}\nRegion: {region}\n"
        f"{region_note}\n\n"
        f"TEXT:\n{content}\n\n"
        "Return JSON with these keys:\n"
        "- red_flags_detected (list of strings)\n"
        "- missing_documentation (list of strings)\n"
        "- verification_steps (list of strings)\n"
        "- investor_safe_reply (string)\n"
        "- jurisdiction_notes (string)\n"
        "- confidence_level ('Low'|'Moderate'|'High')\n"
        "- disclaimer (string, 1–2 lines)\n"
    )

    # Use chat.completions to avoid SDK mismatch issues
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content.strip()

    # try parse JSON; if not JSON, fail safely
    try:
        data = json.loads(text)
    except Exception:
        data = {
            "red_flags_detected": ["Model output was not valid JSON; please try again."],
            "missing_documentation": [],
            "verification_steps": [],
            "investor_safe_reply": "Please share the official documents and verification details.",
            "jurisdiction_notes": region_note or "Verify identity + escrow + contract terms.",
            "confidence_level": "Low",
            "disclaimer": "This is a risk screening tool, not legal advice."
        }
    return data


# =====================
# USAGE LIMITS (5 free)
# =====================
def usage_key(user: Optional[User], client_id: Optional[str]) -> str:
    if user:
        return f"u:{user.id}:scans"
    if client_id:
        # hash so we don't store raw device ID in redis keys
        h = hashlib.sha256(client_id.encode()).hexdigest()[:24]
        return f"c:{h}:scans"
    return "anon:unknown"


def enforce_free_limit(key: str):
    count = int(redis_get(key) or "0")
    if count >= FREE_SCANS:
        raise HTTPException(
            status_code=402,
            detail=f"Free limit reached ({FREE_SCANS} scans). Please sign in or upgrade."
        )


def increment_usage(key: str) -> int:
    return redis_incr(key)


# =====================
# PDF + CHART EXPORT
# =====================
def make_breakdown_chart_png(breakdown: Dict[str, int]) -> bytes:
    labels = list(breakdown.keys())
    values = [breakdown[k] for k in labels]

    fig = plt.figure(figsize=(6, 3))
    plt.bar(labels, values)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 25)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_pdf_bytes(title: str, result: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, title)
    y -= 25

    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Generated: {datetime.utcnow().isoformat()} UTC")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"PTR Score: {result.get('ptr_score')}  |  {result.get('risk_level')}")
    y -= 22

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Risk Breakdown:")
    y -= 14

    c.setFont("Helvetica", 10)
    for k, v in (result.get("risk_breakdown") or {}).items():
        c.drawString(55, y, f"- {k}: {v}/25")
        y -= 14

    y -= 6
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Red Flags Detected:")
    y -= 14
    c.setFont("Helvetica", 10)
    for it in (result.get("red_flags_detected") or [])[:10]:
        c.drawString(55, y, f"- {it[:110]}")
        y -= 14

    y -= 6
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Verification Steps:")
    y -= 14
    c.setFont("Helvetica", 10)
    for it in (result.get("verification_steps") or [])[:10]:
        c.drawString(55, y, f"- {it[:110]}")
        y -= 14

    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Suggested Safe Reply:")
    y -= 14
    c.setFont("Helvetica", 10)
    reply = (result.get("investor_safe_reply") or "")[:700]
    # simple wrap
    words = reply.split()
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > 95:
            c.drawString(55, y, line)
            y -= 14
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        c.drawString(55, y, line)
        y -= 14

    y -= 10
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(40, y, result.get("disclaimer") or "This tool provides risk screening only; verify independently.")
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()


# =====================
# MAIN ANALYZE
# =====================
@app.post("/analyze")
def analyze(
    body: AnalyzeIn,
    x_client_id: Optional[str] = Header(default=None),
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    key = usage_key(user, x_client_id)
    enforce_free_limit(key)

    text = body.content.strip()
    breakdown = heuristic_risk_breakdown(text)
    score = calc_ptr_score(breakdown)
    lvl = risk_level(score)

    memory = build_user_memory(db, user, x_client_id)
    llm = openai_generate_structured(text, body.category, body.region, memory)

    result = {
        "ptr_score": score,
        "risk_level": lvl,
        "risk_breakdown": breakdown,
        "red_flags_detected": llm.get("red_flags_detected", []),
        "missing_documentation": llm.get("missing_documentation", []),
        "verification_steps": llm.get("verification_steps", []),
        "investor_safe_reply": llm.get("investor_safe_reply", ""),
        "jurisdiction_notes": llm.get("jurisdiction_notes", ""),
        "confidence_level": llm.get("confidence_level", "Moderate"),
        "disclaimer": llm.get("disclaimer", "This is risk screening only, not legal advice."),
    }

    # Save scan (for personalization memory)
    s = Scan(
        user_id=user.id if user else None,
        client_id=x_client_id,
        category=body.category,
        input_text=text,
        result_json=json.dumps(result),
    )
    db.add(s)
    db.commit()
    db.refresh(s)

    used = increment_usage(key)
    free_left = max(0, FREE_SCANS - used)

    return {
        **result,
        "scan_id": s.id,
        "free_uses_left": free_left,
        "usage_info": {
            "free_scans_total": FREE_SCANS,
            "free_scans_used": used,
            "identifier_type": "user" if user else "anonymous_device",
        }
    }


@app.post("/feedback")
def submit_feedback(
    body: FeedbackIn,
    x_client_id: Optional[str] = Header(default=None),
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    fb = Feedback(
        scan_id=body.scan_id,
        user_id=user.id if user else None,
        client_id=x_client_id,
        rating=body.rating,
        note=body.note,
    )
    db.add(fb)
    db.commit()
    return {"ok": True}


@app.get("/export/pdf/{scan_id}")
def export_pdf(
    scan_id: int,
    x_client_id: Optional[str] = Header(default=None),
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = db.query(Scan).filter(Scan.id == scan_id).first()
    if not s:
        raise HTTPException(404, "Scan not found")

    # Access control: user owns it OR anonymous client id matches
    if user and s.user_id != user.id:
        raise HTTPException(403, "Not allowed")
    if not user and s.client_id and x_client_id and s.client_id != x_client_id:
        raise HTTPException(403, "Not allowed")
    if not user and not x_client_id:
        raise HTTPException(403, "Not allowed")

    result = json.loads(s.result_json)
    pdf_bytes = build_pdf_bytes("DeedSense AI — Property Trust Report", result)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="deedsense-report-{scan_id}.pdf"'}
    )


@app.get("/export/chart/{scan_id}")
def export_chart(
    scan_id: int,
    x_client_id: Optional[str] = Header(default=None),
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = db.query(Scan).filter(Scan.id == scan_id).first()
    if not s:
        raise HTTPException(404, "Scan not found")

    if user and s.user_id != user.id:
        raise HTTPException(403, "Not allowed")
    if not user and s.client_id and x_client_id and s.client_id != x_client_id:
        raise HTTPException(403, "Not allowed")
    if not user and not x_client_id:
        raise HTTPException(403, "Not allowed")

    result = json.loads(s.result_json)
    png = make_breakdown_chart_png(result.get("risk_breakdown") or {})
    return Response(content=png, media_type="image/png")
