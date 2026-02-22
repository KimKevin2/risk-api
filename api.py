from __future__ import annotations

import os
import time
import math
import json
import uuid
import logging
from typing import Any, Dict, Tuple, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# =========================
# App / Config
# =========================
APP_NAME = os.environ.get("APP_NAME", "RiskNode API")
APP_VERSION = os.environ.get("APP_VERSION", "4.0")

app = FastAPI(title=APP_NAME, version=APP_VERSION)

# NOTE: API_KEY currently unused in your code but kept for compatibility.
API_KEY = os.environ.get("API_KEY", "")

# timeouts / cache
DEX_TIMEOUT_SEC = float(os.environ.get("DEX_TIMEOUT_SEC", "8"))
CACHE_TTL_SEC = int(os.environ.get("CACHE_TTL_SEC", "60"))

# rate limiting (simple, in-memory)
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))  # per IP
RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "30"))       # allow short burst

# logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("risknode")


# =========================
# Response envelope helpers
# =========================
def ok(data: Any, request_id: str):
    return {"ok": True, "request_id": request_id, "data": data}

def fail(
    request_id: str,
    code: str,
    message: str,
    http_status: int,
    retryable: bool = False,
    details: Any = None,
):
    payload = {
        "ok": False,
        "request_id": request_id,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }
    if details is not None:
        payload["error"]["details"] = details
    return JSONResponse(status_code=http_status, content=payload)


# =========================
# Middleware: request_id + basic access log + rate limit
# =========================
_rate_state: Dict[str, Dict[str, float]] = {}  # ip -> {tokens, last_ts}

def _client_ip(req: Request) -> str:
    # If behind proxy/load balancer, Render often sets x-forwarded-for
    xff = req.headers.get("x-forwarded-for")
    if xff:
        # first IP is original client
        return xff.split(",")[0].strip()
    return (req.client.host if req.client else "unknown")

def _rate_allow(ip: str) -> Tuple[bool, float]:
    """Token bucket: returns (allowed, retry_after_seconds)."""
    now = time.time()
    st = _rate_state.get(ip)
    if st is None:
        st = {"tokens": float(RATE_LIMIT_BURST), "last_ts": now}
        _rate_state[ip] = st

    # refill tokens
    elapsed = max(0.0, now - st["last_ts"])
    st["last_ts"] = now
    refill_rate_per_sec = RATE_LIMIT_PER_MIN / 60.0
    st["tokens"] = min(float(RATE_LIMIT_BURST), st["tokens"] + elapsed * refill_rate_per_sec)

    if st["tokens"] >= 1.0:
        st["tokens"] -= 1.0
        return True, 0.0

    # compute retry after to reach 1 token
    missing = 1.0 - st["tokens"]
    retry_after = missing / refill_rate_per_sec if refill_rate_per_sec > 0 else 60.0
    return False, max(1.0, retry_after)

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id

    ip = _client_ip(request)
    allowed, retry_after = _rate_allow(ip)
    if not allowed:
        resp = fail(
            request_id=request_id,
            code="RATE_LIMITED",
            message="Too many requests. Please retry later.",
            http_status=429,
            retryable=True,
            details={"retry_after_sec": int(retry_after)},
        )
        resp.headers["Retry-After"] = str(int(retry_after))
        resp.headers["X-Request-Id"] = request_id
        return resp

    start = time.time()
    try:
        response = await call_next(request)
        dur_ms = int((time.time() - start) * 1000)
        response.headers["X-Request-Id"] = request_id
        logger.info(json.dumps({
            "event": "request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": getattr(response, "status_code", None),
            "duration_ms": dur_ms,
            "ip": ip,
        }))
        return response
    except Exception as e:
        dur_ms = int((time.time() - start) * 1000)
        logger.exception(json.dumps({
            "event": "request_exception",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "duration_ms": dur_ms,
            "ip": ip,
            "error": repr(e),
        }))
        raise


# =========================
# Exception handlers (standard error output)
# =========================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:16])
    # FastAPI's HTTPException.detail can be str or dict
    msg = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return fail(
        request_id=request_id,
        code="HTTP_ERROR",
        message=str(msg),
        http_status=exc.status_code,
        retryable=(exc.status_code >= 500),
        details=None if isinstance(exc.detail, str) else exc.detail,
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:16])
    return fail(
        request_id=request_id,
        code="INTERNAL_ERROR",
        message="Unhandled server error.",
        http_status=500,
        retryable=True,
        details={"type": exc.__class__.__name__},
    )


# =========================
# Utilities
# =========================
def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default

def clamp(x, lo, hi):
    return max(lo, min(x, hi))

def calc_pair_age_days(pair_created_at_ms):
    if not pair_created_at_ms:
        return None
    now_ms = int(time.time() * 1000)
    age_days = (now_ms - pair_created_at_ms) / (1000 * 60 * 60 * 24)
    if age_days < 0:
        return None
    return age_days


# =========================
# Chain + address validation
# =========================
SUPPORTED_CHAINS = {
    "ethereum", "base", "arbitrum", "polygon", "bsc", "avalanche", "optimism",
    "solana",
}

CHAIN_ALIASES = {
    "eth": "ethereum",
    "arb": "arbitrum",
    "matic": "polygon",
    "bnb": "bsc",
    "avax": "avalanche",
    "op": "optimism",
}

_BASE58_ALPHABET = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")

def normalize_chain_id(chain_id: str | None) -> str:
    cid = (chain_id or "ethereum").strip().lower()
    cid = CHAIN_ALIASES.get(cid, cid)
    if cid not in SUPPORTED_CHAINS:
        raise HTTPException(status_code=400, detail=f"Unsupported chainId: {cid}")
    return cid

def is_evm_address(addr: str) -> bool:
    a = (addr or "").strip()
    return a.startswith("0x") and len(a) == 42

def is_base58(s: str) -> bool:
    if not s:
        return False
    return all(c in _BASE58_ALPHABET for c in s)

def is_solana_address(addr: str) -> bool:
    a = (addr or "").strip()
    if not (32 <= len(a) <= 44):
        return False
    return is_base58(a)

def validate_contract_for_chain(contract: str, chain_id: str) -> str:
    c = (contract or "").strip()
    if chain_id == "solana":
        if not is_solana_address(c):
            raise HTTPException(status_code=400, detail="Invalid Solana address format.")
        return c
    if not is_evm_address(c):
        raise HTTPException(status_code=400, detail="Invalid EVM contract address format.")
    return c


# =========================
# HTTP client w/ retry
# =========================
_session = requests.Session()
_retry = Retry(
    total=3,
    connect=2,
    read=2,
    status=3,
    backoff_factor=0.4,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=10, pool_maxsize=10)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


# =========================
# TTL cache (token pairs)
# =========================
_cache_pairs: Dict[Tuple[str, str], Tuple[float, Any]] = {}

def _cache_get(key: Tuple[str, str]):
    item = _cache_pairs.get(key)
    if not item:
        return None
    exp, val = item
    if time.time() > exp:
        _cache_pairs.pop(key, None)
        return None
    return val

def _cache_set(key: Tuple[str, str], val: Any, ttl_sec: int):
    _cache_pairs[key] = (time.time() + ttl_sec, val)


# =========================
# Dexscreener data
# =========================
def fetch_token_pairs(chain_id: str, token_address: str):
    # cache (short TTL)
    ck = (chain_id, token_address.lower())
    cached = _cache_get(ck)
    if cached is not None:
        return cached

    url = f"https://api.dexscreener.com/token-pairs/v1/{chain_id}/{token_address}"
    try:
        r = _session.get(url, timeout=DEX_TIMEOUT_SEC)
        # Retry adapter handles transient status; but still check success
        if r.status_code >= 500:
            raise HTTPException(status_code=502, detail="Upstream DexScreener error (5xx).")
        if r.status_code == 429:
            raise HTTPException(status_code=502, detail="Upstream DexScreener rate-limited (429).")
        r.raise_for_status()
        data = r.json()
        _cache_set(ck, data, CACHE_TTL_SEC)
        return data
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="DexScreener timeout.")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"DexScreener request failed: {type(e).__name__}")


def pick_best_pair(pairs):
    def liq_usd(p):
        liq = p.get("liquidity") or {}
        return safe_float(liq.get("usd"), 0)

    pairs_sorted = sorted(pairs, key=liq_usd, reverse=True)
    return pairs_sorted[0] if pairs_sorted else None


def summarize(pair):
    liq = pair.get("liquidity") or {}
    vol = pair.get("volume") or {}
    txns = pair.get("txns") or {}
    pc = pair.get("priceChange") or {}

    h24_txns = txns.get("h24") or {}
    h6_txns = txns.get("h6") or {}
    h1_txns = txns.get("h1") or {}

    return {
        "pair": f'{pair.get("baseToken", {}).get("symbol")} / {pair.get("quoteToken", {}).get("symbol")}',
        "baseSymbol": pair.get("baseToken", {}).get("symbol"),
        "quoteSymbol": pair.get("quoteToken", {}).get("symbol"),
        "dex": pair.get("dexId"),
        "pairAddress": pair.get("pairAddress"),
        "chainId": pair.get("chainId"),
        "priceUsd": safe_float(pair.get("priceUsd")),
        "liquidityUsd": safe_float(liq.get("usd")),
        "volume24h": safe_float(vol.get("h24")),
        "volume6h": safe_float(vol.get("h6")),
        "volume1h": safe_float(vol.get("h1")),
        "buys24h": safe_int(h24_txns.get("buys")),
        "sells24h": safe_int(h24_txns.get("sells")),
        "buys6h": safe_int(h6_txns.get("buys")),
        "sells6h": safe_int(h6_txns.get("sells")),
        "buys1h": safe_int(h1_txns.get("buys")),
        "sells1h": safe_int(h1_txns.get("sells")),
        "priceChange24h": safe_float(pc.get("h24")),
        "priceChange6h": safe_float(pc.get("h6")),
        "priceChange1h": safe_float(pc.get("h1")),
        "fdv": safe_float(pair.get("fdv")),
        "marketCap": safe_float(pair.get("marketCap")),
        "pairCreatedAt": safe_int(pair.get("pairCreatedAt")),
        "url": pair.get("url"),
    }


# =========================
# Risk model (kept mostly same)
# =========================
def score_log_low_liquidity(L):
    if L <= 0:
        return 40
    val = (math.log10(5_000_000) - math.log10(L)) / (math.log10(5_000_000) - math.log10(50_000))
    return clamp(val * 40, 0, 40)

def score_volume_heat(V24, L):
    if L <= 0:
        return 10
    r = V24 / L
    val = (r - 1) / (10 - 1)
    return clamp(val * 25, 0, 25)

def score_sell_pressure(buys, sells):
    total = buys + sells
    if total < 30:
        return 0
    if buys <= 0 and sells > 0:
        return 20
    ratio = sells / max(1, buys)
    val = (ratio - 1) / (6 - 1)
    return clamp(val * 20, 0, 20)

def score_volatility(pc1, pc24):
    s = 0
    if abs(pc1) >= 10:
        s += clamp((abs(pc1) - 10) / (40 - 10) * 10, 0, 10)
    if abs(pc24) >= 20:
        s += clamp((abs(pc24) - 20) / (80 - 20) * 10, 0, 10)
    return clamp(s, 0, 20)

def score_fdv_gap(fdv, L):
    if fdv <= 0 or L <= 0:
        return 0
    r = fdv / L
    val = (r - 50) / (500 - 50)
    return clamp(val * 15, 0, 15)

def score_new_pair(age_days):
    if age_days is None:
        return 0
    if age_days <= 0:
        return 10
    val = (14 - age_days) / 14
    return clamp(val * 10, 0, 10)

def recommend_action(risk_score, liquidity_usd):
    if risk_score >= 51:
        return "AVOID"
    if risk_score >= 21:
        return "WAIT"
    if liquidity_usd < 200_000:
        return "WAIT"
    return "OK"

def recommended_position_pct(action: str, risk_score: int):
    if action == "AVOID":
        return 0.0
    if action == "WAIT":
        return 0.8 if risk_score <= 30 else 0.5
    return 2.0 if risk_score <= 10 else 1.5

def suggested_slippage_bps(liquidity_usd: float, risk_score: int):
    bps = 30
    if liquidity_usd < 200_000:
        bps += 70
    elif liquidity_usd < 1_000_000:
        bps += 40
    elif liquidity_usd < 5_000_000:
        bps += 20

    if risk_score >= 51:
        bps += 80
    elif risk_score >= 21:
        bps += 30

    return int(clamp(bps, 30, 300))

def max_order_usd(liquidity_usd: float, action: str):
    if liquidity_usd <= 0:
        return 0.0
    if action == "AVOID":
        return 0.0
    frac = 0.001 if action == "WAIT" else 0.003
    return round(liquidity_usd * frac, 2)


def honeypot_heuristic(buys24: int, sells24: int, volume24: float, liquidity: float,
                       pc1: float | None, pc24: float | None):

    signals = []
    score = 0

    b = max(0, int(buys24 or 0))
    s = max(0, int(sells24 or 0))
    V = float(volume24 or 0.0)
    L = float(liquidity or 0.0)
    pc1 = float(pc1 or 0.0)
    pc24 = float(pc24 or 0.0)

    if b + s < 30:
        signals.append("거래 샘플 적음(판단 유보)")
        score += 2
        level = "LOW"
        return False, score, signals, level

    ratio = s / max(1, b)

    if ratio < 0.10 and b >= 80:
        score += 18
        signals.append(f"매수 대비 매도 비율 매우 낮음 (sells/buys={ratio:.2f})")
    elif ratio < 0.15 and b >= 50:
        score += 12
        signals.append(f"매수 대비 매도 비율 낮음 (sells/buys={ratio:.2f})")

    if L > 0:
        heat = V / L
        if heat > 3.0 and ratio < 0.15:
            score += 6
            signals.append(f"과열 구간에서 매도 부족 (vol/liquidity={heat:.2f})")

    if pc1 > 8 and ratio < 0.15:
        score += 4
        signals.append(f"단기 급등인데 매도 부족 (1h={pc1:+.1f}%)")

    if pc24 > 40 and ratio < 0.20:
        score += 4
        signals.append(f"24h 급등인데 매도 부족 (24h={pc24:+.1f}%)")

    score = min(25, score)

    if score >= 18:
        level = "SUSPECTED"
    elif score >= 10:
        level = "WATCH"
    else:
        level = "LOW"

    suspected = (level == "SUSPECTED")
    return suspected, score, signals, level


def analyze_token(contract: str, chain_id: str = "ethereum"):
    pairs = fetch_token_pairs(chain_id, contract)
    if not pairs:
        raise HTTPException(status_code=404, detail="No DEX pairs found for this token on this chain.")

    best = pick_best_pair(pairs)
    if not best:
        raise HTTPException(status_code=404, detail="No viable DEX pair found.")

    s = summarize(best)

    L = s["liquidityUsd"]
    V24 = s["volume24h"]
    buys24 = s["buys24h"]
    sells24 = s["sells24h"]
    pc1 = s["priceChange1h"]
    pc24 = s["priceChange24h"]
    fdv = s["fdv"]
    age_days = calc_pair_age_days(s.get("pairCreatedAt"))

    p1 = score_log_low_liquidity(L)
    p2 = score_volume_heat(V24, L)
    p3 = score_sell_pressure(buys24, sells24)
    p4 = score_volatility(pc1, pc24)
    p5 = score_fdv_gap(fdv, L)
    p6 = score_new_pair(age_days)

    risk = int(round(clamp(p1 + p2 + p3 + p4 + p5 + p6, 0, 100)))

    honeypot_suspected, honeypot_score, honeypot_signals, honeypot_level = honeypot_heuristic(
        buys24, sells24, V24, L, pc1, pc24
    )
    risk = min(100, risk + honeypot_score)

    if risk <= 20:
        level = "LOW"
    elif risk <= 50:
        level = "MEDIUM"
    else:
        level = "HIGH"

    components = {
        "liquidityRisk": int(round(p1)),
        "heatRisk": int(round(p2)),
        "sellPressureRisk": int(round(p3)),
        "volatilityRisk": int(round(p4)),
        "fdvGapRisk": int(round(p5)),
        "newPairRisk": int(round(p6)),
        "honeypotRisk": honeypot_score,
    }

    reasons = []
    if components["liquidityRisk"] >= 10:
        reasons.append(f"유동성 낮음 (${L:,.0f})")
    if components["fdvGapRisk"] >= 6 and L > 0:
        reasons.append(f"FDV 대비 유동성 낮음 (fdv/liquidity={fdv/L:.0f}x)")
    if components["heatRisk"] >= 10 and L > 0:
        reasons.append(f"거래 과열 가능성 (vol/liquidity={V24/L:.2f})")
    if components["sellPressureRisk"] >= 10:
        reasons.append(f"매도 압력 (sells/buys={sells24/max(1,buys24):.2f})")
    if components["volatilityRisk"] >= 10:
        reasons.append(f"급격한 변동 (1h={pc1:+.1f}%, 24h={pc24:+.1f}%)")
    if components["newPairRisk"] >= 6 and age_days is not None:
        reasons.append(f"신생 페어 (age={age_days:.1f}일)")
    if not reasons:
        reasons = ["유동성 충분", "단기 변동성 안정", "거래 활동 정상 범위"]

    action = recommend_action(risk, L)
    pos_pct = recommended_position_pct(action, risk)
    slippage_bps = suggested_slippage_bps(L, risk)
    max_usd = max_order_usd(L, action)

    summary = (
        f"{level} risk ({risk}/100) | action={action} | "
        f"maxPosition={pos_pct}% | maxOrder≈${max_usd:,.0f} | slippage≈{slippage_bps}bps"
    )

    guidance = {
        "action": action,
        "maxPositionPct": pos_pct,
        "maxOrderUsd": max_usd,
        "suggestedSlippageBps": slippage_bps,
    }

    # confidence heuristic (simple): higher liquidity and more txns => higher confidence
    txn24 = (s.get("buys24h", 0) + s.get("sells24h", 0))
    conf = 0.55
    if L >= 1_000_000:
        conf += 0.15
    elif L >= 200_000:
        conf += 0.08
    if txn24 >= 300:
        conf += 0.15
    elif txn24 >= 80:
        conf += 0.08
    if honeypot_level in ("WATCH", "SUSPECTED"):
        conf -= 0.05
    confidence = float(clamp(conf, 0.35, 0.95))

    return {
        "riskScore": risk,
        "riskLevel": level,
        "confidence": confidence,
        "summary": summary,
        "reasons": reasons[:8],
        "components": components,
        "guidance": guidance,
        "pair": s["pair"],
        "dex": s["dex"],
        "liquidityUsd": s["liquidityUsd"],
        "volume24h": s["volume24h"],
        "buys24h": s["buys24h"],
        "sells24h": s["sells24h"],
        "priceChange1h": s["priceChange1h"],
        "priceChange24h": s["priceChange24h"],
        "fdv": s["fdv"],
        "url": s["url"],
        "honeypotSuspected": honeypot_suspected,
        "honeypotScore": honeypot_score,
        "honeypotSignals": honeypot_signals,
        "honeypotLevel": honeypot_level,
    }


# =========================
# API schema
# =========================
class AnalyzeRequest(BaseModel):
    contract: str
    chainId: str | None = "ethereum"


# =========================
# Health / Ready
# =========================
@app.get("/health")
def health(request: Request):
    rid = request.state.request_id
    return ok({"status": "ok"}, rid)

@app.get("/ready")
def ready(request: Request):
    rid = request.state.request_id
    # lightweight readiness: config present + session configured
    checks = {
        "api_key_set": bool(API_KEY),
        "dex_timeout_sec": DEX_TIMEOUT_SEC,
        "cache_ttl_sec": CACHE_TTL_SEC,
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
    }
    # If API_KEY is not required, don't fail readiness based on it.
    return ok({"ready": True, "checks": checks}, rid)


# =========================
# Endpoints (keep backward compatibility)
# =========================
@app.post("/analyze")
def analyze(req: AnalyzeRequest, request: Request):
    rid = request.state.request_id
    chain_id = normalize_chain_id(req.chainId)
    contract = validate_contract_for_chain(req.contract, chain_id)
    data = analyze_token(contract, chain_id)
    return ok(data, rid)

@app.post("/quick-scan")
def quick_scan(req: AnalyzeRequest, request: Request):
    rid = request.state.request_id
    chain_id = normalize_chain_id(req.chainId)
    contract = validate_contract_for_chain(req.contract, chain_id)

    result = analyze_token(contract, chain_id)
    safe = result["riskScore"] <= 20 and not result["honeypotSuspected"]

    data = {
        "safeToTrade": safe,
        "riskScore": result["riskScore"],
        "riskLevel": result["riskLevel"],
        "confidence": result["confidence"],
        "recommendation": "avoid" if result["guidance"]["action"] == "AVOID" else ("caution" if result["guidance"]["action"] == "WAIT" else "buy"),
        "action": result["guidance"]["action"],
        "maxOrderUsd": result["guidance"]["maxOrderUsd"],
        "slippageBps": result["guidance"]["suggestedSlippageBps"],
    }
    return ok(data, rid)


# --- ACP ultra-light endpoint (seller-facing) ---
# Keep existing response keys EXACT to avoid breaking ACP integration.
@app.post("/acp/crypto_quick_scan")
def acp_crypto_quick_scan(req: AnalyzeRequest, request: Request):
    chain_id = normalize_chain_id(req.chainId)
    contract = validate_contract_for_chain(req.contract, chain_id)

    result = analyze_token(contract, chain_id)
    safe = (result["riskScore"] <= 20) and (not result.get("honeypotSuspected", False))

    return {
        "safeToTrade": safe,
        "riskScore": result["riskScore"],
        "action": result["guidance"]["action"],
    }


# --- ACP v2 (AI-friendly decision JSON; optional new endpoint) ---
@app.post("/acp/crypto_quick_scan_v2")
def acp_crypto_quick_scan_v2(req: AnalyzeRequest, request: Request):
    rid = request.state.request_id
    chain_id = normalize_chain_id(req.chainId)
    contract = validate_contract_for_chain(req.contract, chain_id)

    result = analyze_token(contract, chain_id)
    safe = (result["riskScore"] <= 20) and (not result.get("honeypotSuspected", False))

    action = result["guidance"]["action"]
    recommendation = "avoid" if action == "AVOID" else ("caution" if action == "WAIT" else "buy")

    data = {
        "safeToTrade": safe,
        "risk_score": result["riskScore"],
        "risk_level": result["riskLevel"].lower(),
        "confidence": result["confidence"],
        "recommendation": recommendation,
        "components": result["components"],
        "signals": {
            "honeypot_suspected": bool(result.get("honeypotSuspected")),
            "honeypot_level": result.get("honeypotLevel"),
        },
    }
    return ok(data, rid)