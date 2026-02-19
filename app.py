import os
import json
import time
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor

import jwt  # PyJWT
import stripe
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------
# Config
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
DEFAULT_FREE_SCANS = int(os.getenv("DEFAULT_FREE_SCANS", "5"))

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_PRICE_PRO_MONTHLY = os.getenv("STRIPE_PRICE_PRO_MONTHLY")
STRIPE_PRICE_PRO_YEARLY = os.getenv("STRIPE_PRICE_PRO_YEARLY")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

app = FastAPI(title="DeedSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# DB Helpers
# -----------------------------
def db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    # Safe "create if not exists" schema
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            create table if not exists subscriptions (
              user_id text primary key,
              plan text not null default 'free',
              stripe_customer_id text,
              stripe_subscription_id text,
              status text default 'active',
              current_period_end timestamptz
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
              result_json jsonb not null
            );

            create table if not exists api_keys (
              id uuid default gen_random_uuid() primary key,
              user_id text not null,
              api_key_hash text not null,
              label text,
              active boolean not null default true
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
    if not auth_header:
        return None
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.replace("Bearer ", "").strip()
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="SUPABASE_JWT_SECRET not configured")

    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        # Supabase user id is usually in "sub"
        return {"user_id": payload.get("sub"), "email": payload.get("email"), "role": payload.get("role")}
    except Exception:
        return None


def month_key_now() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.year:04d}-{now.month:02d}"


def get_subscription(user_id: str) -> Dict[str, Any]:
    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("select * from subscriptions where user_id=%s", (user_id,))
            row = cur.fetchone()
            if not row:
                # create default free
                cur.execute(
                    "insert into subscriptions (user_id, plan, status) values (%s, 'free', 'active')",
                    (user_id,),
                )
                conn.commit()
                return {"user_id": user_id, "plan": "free", "status": "active"}
            return dict(row)


def increment_usage(user_id: str) -> int:
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


def get_usage(user_id: str) -> int:
    mk = month_key_now()
    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("select scans_used from usage_monthly where user_id=%s and month=%s", (user_id, mk))
            row = cur.fetchone()
            return int(row["scans_used"]) if row else 0


def check_ent_api_key(x_api_key: Optional[str]) -> Optional[str]:
    # Enterprise integration can use API keys without Supabase login.
    if not x_api_key:
        return None
    # Hash it
    digest = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()
    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "select user_id from api_keys where api_key_hash=%s and active=true limit 1",
                (digest,),
            )
            row = cur.fetchone()
            return row["user_id"] if row else None


# -----------------------------
# Models
# -----------------------------
class AnalyzeIn(BaseModel):
    text: str


# -----------------------------
# Simple “scanner” placeholder
# (Swap later with your AI model)
# -----------------------------
def basic_risk_scan(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    signals = []
    score = 0

    def hit(phrase, points, reason):
        nonlocal score
        if phrase in t:
            signals.append(reason)
            score += points

    hit("limited time", 15, "Urgency pressure detected")
    hit("last unit", 15, "Artificial scarcity language detected")
    hit("guaranteed returns", 20, "Suspicious guarantee claim detected")
    hit("no questions asked", 10, "High-pressure reassurance detected")
    hit("book now", 10, "Call-to-action pressure detected")

    score = min(score, 100)
    label = "Low" if score < 20 else "Medium" if score < 50 else "High"

    return {
        "risk_score": score,
        "risk_label": label,
        "signals": signals,
        "summary": "AI-like risk signal summary (MVP placeholder). Replace with model output.",
        "confidence": 0.72,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/me")
def me(authorization: Optional[str] = Header(default=None)):
    u = get_user_from_bearer(authorization)
    if not u or not u.get("user_id"):
        return {"signed_in": False}
    sub = get_subscription(u["user_id"])
    usage = get_usage(u["user_id"])
    return {"signed_in": True, "user": u, "subscription": sub, "usage_month": usage, "free_limit": DEFAULT_FREE_SCANS}


@app.post("/analyze")
def analyze(payload: AnalyzeIn, authorization: Optional[str] = Header(default=None), x_api_key: Optional[str] = Header(default=None)):
    # Allow enterprise API key OR signed-in token
    user_id = check_ent_api_key(x_api_key)
    if not user_id:
        u = get_user_from_bearer(authorization)
        if not u or not u.get("user_id"):
            raise HTTPException(status_code=401, detail="Sign in required (or Enterprise API key).")
        user_id = u["user_id"]

    sub = get_subscription(user_id)
    plan = (sub.get("plan") or "free").lower()
    status = (sub.get("status") or "active").lower()

    if plan != "free" and status not in ["active", "trialing"]:
        raise HTTPException(status_code=402, detail="Subscription not active. Please renew to continue.")

    if plan == "free":
        used = get_usage(user_id)
        if used >= DEFAULT_FREE_SCANS:
            raise HTTPException(status_code=402, detail=f"Free scans limit reached ({DEFAULT_FREE_SCANS}/month). Upgrade to Pro for unlimited scans.")
        new_used = increment_usage(user_id)
    else:
        new_used = get_usage(user_id)

    result = basic_risk_scan(payload.text)

    # Save history only for signed-in users (not for API-key-only if you prefer).
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "insert into scan_history (user_id, input_type, filename, result_json) values (%s, %s, %s, %s)",
                (user_id, "text", None, json.dumps(result)),
            )
        conn.commit()

    return {"plan": plan, "usage_month": new_used, "result": result}


@app.get("/history")
def history(authorization: Optional[str] = Header(default=None)):
    u = get_user_from_bearer(authorization)
    if not u or not u.get("user_id"):
        raise HTTPException(status_code=401, detail="Sign in required.")
    user_id = u["user_id"]

    sub = get_subscription(user_id)
    plan = (sub.get("plan") or "free").lower()
    if plan == "free":
        # free has no history
        return {"items": [], "note": "History available on Pro/Enterprise."}

    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "select id, created_at, input_type, filename, result_json from scan_history where user_id=%s order by created_at desc limit 50",
                (user_id,),
            )
            rows = cur.fetchall()
            return {"items": rows}


# -----------------------------
# Stripe Billing
# -----------------------------
@app.post("/billing/create-checkout-session")
async def create_checkout_session(request: Request, authorization: Optional[str] = Header(default=None)):
    u = get_user_from_bearer(authorization)
    if not u or not u.get("user_id"):
        raise HTTPException(status_code=401, detail="Sign in required.")
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="Stripe not configured.")

    body = await request.json()
    billing = body.get("billing", "monthly")  # monthly/yearly

    price_id = STRIPE_PRICE_PRO_MONTHLY if billing == "monthly" else STRIPE_PRICE_PRO_YEARLY
    if not price_id:
        raise HTTPException(status_code=500, detail="Stripe price ID not configured for selected billing.")

    user_id = u["user_id"]

    # create/find stripe customer
    sub = get_subscription(user_id)
    customer_id = sub.get("stripe_customer_id")

    if not customer_id:
        customer = stripe.Customer.create(
            email=u.get("email"),
            metadata={"user_id": user_id},
        )
        customer_id = customer["id"]
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("update subscriptions set stripe_customer_id=%s where user_id=%s", (customer_id, user_id))
            conn.commit()

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=body.get("success_url", "https://deedsense-ui.onrender.com/?success=1"),
        cancel_url=body.get("cancel_url", "https://deedsense-ui.onrender.com/?canceled=1"),
        metadata={"user_id": user_id, "plan": "pro"},
    )
    return {"url": session.url}


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret not configured.")
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook signature.")

    # Handle subscription events
    etype = event["type"]
    data = event["data"]["object"]

    def upsert_subscription(user_id: str, plan: str, status: str, sub_id: Optional[str], period_end: Optional[int]):
        ts = datetime.fromtimestamp(period_end, tz=timezone.utc) if period_end else None
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    insert into subscriptions (user_id, plan, status, stripe_subscription_id, current_period_end)
                    values (%s, %s, %s, %s, %s)
                    on conflict (user_id)
                    do update set plan=excluded.plan, status=excluded.status, stripe_subscription_id=excluded.stripe_subscription_id, current_period_end=excluded.current_period_end
                """, (user_id, plan, status, sub_id, ts))
            conn.commit()

    if etype in ["checkout.session.completed"]:
        user_id = data.get("metadata", {}).get("user_id")
        plan = data.get("metadata", {}).get("plan", "pro")
        # Subscription created after checkout
        # We'll rely on subscription.updated events too.
        if user_id:
            upsert_subscription(user_id, plan, "active", None, None)

    if etype in ["customer.subscription.updated", "customer.subscription.created"]:
        sub_id = data.get("id")
        status = data.get("status")
        period_end = data.get("current_period_end")
        user_id = (data.get("metadata") or {}).get("user_id")

        # If metadata absent, try customer metadata
        if not user_id and data.get("customer"):
            cust = stripe.Customer.retrieve(data["customer"])
            user_id = (cust.get("metadata") or {}).get("user_id")

        if user_id:
            plan = "pro"  # map price IDs if you want multiple pro tiers
            upsert_subscription(user_id, plan, status, sub_id, period_end)

    if etype in ["customer.subscription.deleted"]:
        sub_id = data.get("id")
        status = "canceled"
        user_id = (data.get("metadata") or {}).get("user_id")
        if not user_id and data.get("customer"):
            cust = stripe.Customer.retrieve(data["customer"])
            user_id = (cust.get("metadata") or {}).get("user_id")
        if user_id:
            upsert_subscription(user_id, "free", status, sub_id, None)

    return {"received": True}


# -----------------------------
# Admin (simple)
# Admin is defined by Supabase JWT "role" claim == "admin"
# -----------------------------
@app.get("/admin/users")
def admin_users(authorization: Optional[str] = Header(default=None)):
    u = get_user_from_bearer(authorization)
    if not u or u.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")
    with db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("select user_id, plan, status, current_period_end from subscriptions order by user_id asc limit 200")
            return {"items": cur.fetchall()}
