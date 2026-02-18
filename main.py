import requests
import time
import math


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


def fetch_token_pairs(chain_id: str, token_address: str):
    url = f"https://api.dexscreener.com/token-pairs/v1/{chain_id}/{token_address}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


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


def calc_pair_age_days(pair_created_at_ms):
    if not pair_created_at_ms:
        return None
    now_ms = int(time.time() * 1000)
    age_days = (now_ms - pair_created_at_ms) / (1000 * 60 * 60 * 24)
    if age_days < 0:
        return None
    return age_days


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


# ----- Risk component scoring (0~100) -----

def score_log_low_liquidity(L):
    # 0~40
    if L <= 0:
        return 40
    val = (math.log10(5_000_000) - math.log10(L)) / (math.log10(5_000_000) - math.log10(50_000))
    return clamp(val * 40, 0, 40)


def score_volume_heat(V24, L):
    # 0~25
    if L <= 0:
        return 10
    r = V24 / L
    val = (r - 1) / (10 - 1)
    return clamp(val * 25, 0, 25)


def score_sell_pressure(buys, sells):
    # 0~20
    total = buys + sells
    if total < 30:
        return 0
    if buys <= 0 and sells > 0:
        return 20
    ratio = sells / max(1, buys)
    val = (ratio - 1) / (6 - 1)
    return clamp(val * 20, 0, 20)


def score_volatility(pc1, pc24):
    # 0~20
    s = 0
    if abs(pc1) >= 10:
        s += clamp((abs(pc1) - 10) / (40 - 10) * 10, 0, 10)
    if abs(pc24) >= 20:
        s += clamp((abs(pc24) - 20) / (80 - 20) * 10, 0, 10)
    return clamp(s, 0, 20)


def score_fdv_gap(fdv, L):
    # 0~15
    if fdv <= 0 or L <= 0:
        return 0
    r = fdv / L
    val = (r - 50) / (500 - 50)
    return clamp(val * 15, 0, 15)


def score_new_pair(age_days):
    # 0~10
    if age_days is None:
        return 0
    if age_days <= 0:
        return 10
    val = (14 - age_days) / 14
    return clamp(val * 10, 0, 10)


def recommend_action(risk_score, liquidity_usd):
    # 매출용 룰: LOW=OK, MEDIUM=WAIT, HIGH=AVOID
    if risk_score >= 51:
        return "AVOID"
    if risk_score >= 21:
        return "WAIT"
    if liquidity_usd < 200_000:
        return "WAIT"
    return "OK"


# ----- NEW: Trading guidance (position / order / slippage) -----

def recommended_position_pct(action: str, risk_score: int):
    """
    매우 단순하고 보수적인 가이드.
    (에이전트가 바로 쓸 수 있게)
    """
    if action == "AVOID":
        return 0.0
    if action == "WAIT":
        # 리스크가 MEDIUM이라도 낮으면 0.5~1.0
        if risk_score <= 30:
            return 0.8
        return 0.5
    # OK
    if risk_score <= 10:
        return 2.0
    return 1.5


def suggested_slippage_bps(liquidity_usd: float, risk_score: int):
    """
    유동성 낮을수록, 리스크 높을수록 슬리피지 넉넉히.
    bps = 1/100%
    """
    bps = 30  # 기본 0.30%
    if liquidity_usd < 200_000:
        bps += 70  # +0.70%
    elif liquidity_usd < 1_000_000:
        bps += 40  # +0.40%
    elif liquidity_usd < 5_000_000:
        bps += 20  # +0.20%

    if risk_score >= 51:
        bps += 80
    elif risk_score >= 21:
        bps += 30

    return int(clamp(bps, 30, 300))  # 0.30%~3.00%


def max_order_usd(liquidity_usd: float, action: str):
    """
    아주 실전적인 규칙:
    - 풀 유동성의 x% 이상 주문 넣으면 가격 밀릴 확률 ↑
    """
    if liquidity_usd <= 0:
        return 0.0
    if action == "AVOID":
        return 0.0
    if action == "WAIT":
        frac = 0.002  # 0.2%
    else:
        frac = 0.005  # 0.5%
    return round(liquidity_usd * frac, 2)


def calculate_risk(s):
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
    }

    # 이유(설명) - 짧게, 상위만
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

    return risk, level, reasons[:8], components, summary, guidance


if __name__ == "__main__":
    print("=== AI Token Risk Engine v3.2 ===")
    token = input("토큰 컨트랙트 주소(0x로 시작): ").strip()
    chain_id = "ethereum"

    try:
        pairs = fetch_token_pairs(chain_id, token)
        if not pairs:
            print("❌ DEX에 페어가 없습니다.")
            raise SystemExit(1)

        best = pick_best_pair(pairs)
        s = summarize(best)

        risk_score, risk_level, reasons, components, summary, guidance = calculate_risk(s)

        report = {
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "summary": summary,
            "reasons": reasons,
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
        }

        print("\n=== Risk Report ===")
        print(report)

    except Exception as e:
        print("에러 발생:", e)