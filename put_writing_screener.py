
"""
Put-Writing Candidate Screener
================================
Screens for stocks down 5%+ intraday, then finds far-OTM put option
candidates suitable for cash-secured put writing.

Stock screen:
    - Down >= 5% since market open
    - Market cap >= $5B
    - Avg daily volume >= 1,000,000 shares
    - No earnings report in the next 7 days

Options screen (per passing stock):
    - Open interest > 1,000 contracts
    - Expiration <= 180 days out
    - |Delta| < 0.10 (computed via Black-Scholes-Merton, since yfinance
      does not return delta directly -- only implied volatility)

Delivery:
    - Emails results via Gmail SMTP (same pattern as the existing
      5%-drop notifier). Intended to run on a schedule via GitHub Actions.

Data source: yfinance (free, no API key required)
"""

import math
import os
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

import requests
import yfinance as yf
from scipy.stats import norm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ALL_TICKERS_CSV_URL = (
    "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/"
    "main/tickers/all.csv"
)

TOP_N_BY_MARKET_CAP = 1500

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    "JPM", "V", "UNH", "XOM", "MA", "HD", "PG", "COST", "MRK", "ABBV",
    "CVX", "PEP", "KO", "WMT", "BAC", "DIS", "ADBE", "CRM", "NFLX",
    "AMD", "INTC", "QCOM", "TXN", "HON", "UPS", "CAT", "GS", "MS",
]


def get_top_market_cap_tickers(top_n=TOP_N_BY_MARKET_CAP):
    """
    Fetch a daily-updated, market-cap-sorted list of US-listed tickers
    live from GitHub (NASDAQ Screener API-sourced) and return the top N
    by market cap. Column names/positions are looked up defensively since
    the exact schema of this community-maintained file isn't guaranteed
    to stay identical over time.
    Falls back to a small static large-cap list if the fetch or parse fails.
    """
    try:
        resp = requests.get(ALL_TICKERS_CSV_URL, timeout=20)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        if not lines:
            raise ValueError("Empty response from ticker list source")

        header = [h.strip().strip('"') for h in lines[0].split(",")]

        symbol_col_candidates = ["Symbol", "symbol", "Ticker", "ticker"]
        symbol_idx = next(
            (header.index(c) for c in symbol_col_candidates if c in header),
            None,
        )
        if symbol_idx is None:
            raise ValueError(
                f"Could not find a symbol/ticker column in header: {header}"
            )

        tickers = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = [p.strip().strip('"') for p in line.split(",")]
            if len(parts) <= symbol_idx:
                continue
            symbol = parts[symbol_idx].replace(".", "-")
            if symbol:
                tickers.append(symbol)

        if len(tickers) < 1000:
            print(f"[warn] Only parsed {len(tickers)} tickers, "
                  f"falling back to static list.")
            return FALLBACK_TICKERS

        top_tickers = tickers[:top_n]
        print(f"Fetched {len(tickers)} total tickers, "
              f"using top {len(top_tickers)} by market cap.")
        return top_tickers

    except Exception as e:
        print(f"[warn] Failed to fetch market-cap-ranked ticker list ({e}), "
              f"falling back to static list.")
        return FALLBACK_TICKERS


TICKER_UNIVERSE = get_top_market_cap_tickers()

DROP_THRESHOLD_PCT = 5.0
MIN_MARKET_CAP = 5_000_000_000
MIN_AVG_VOLUME = 1_000_000
EARNINGS_EXCLUSION_DAYS = 7

MIN_OPEN_INTEREST = 1000
MAX_DAYS_TO_EXPIRATION = 180
MAX_ABS_DELTA = 0.10

RISK_FREE_RATE = 0.045

CAPITAL_PER_POSITION = 50_000
MIN_TOTAL_PREMIUM = 1000

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", GMAIL_USER)


# ---------------------------------------------------------------------------
# BLACK-SCHOLES-MERTON DELTA (dividend-adjusted)
# ---------------------------------------------------------------------------

def bsm_put_delta(S, K, T, r, sigma, q):
    """
    Black-Scholes-Merton put option delta (dividend-adjusted).
    Returns put delta, which ranges from 0 to -1.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    put_delta = math.exp(-q * T) * (norm.cdf(d1) - 1)
    return put_delta


# ---------------------------------------------------------------------------
# STOCK SCREEN
# ---------------------------------------------------------------------------

def get_dividend_yield(ticker_obj, info):
    """Best-effort dividend yield extraction, defaults to 0 if unavailable."""
    div_yield = info.get("dividendYield")
    if div_yield is None:
        return 0.0
    return div_yield / 100 if div_yield > 1 else div_yield


def has_earnings_soon(ticker_obj, days=EARNINGS_EXCLUSION_DAYS):
    """Returns True if an earnings date falls within the next `days` days."""
    try:
        cal = ticker_obj.get_earnings_dates(limit=4)
        if cal is None or cal.empty:
            return False
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days)
        for dt in cal.index:
            dt_utc = dt.tz_convert("UTC") if dt.tzinfo else dt.tz_localize("UTC")
            if now <= dt_utc <= cutoff:
                return True
        return False
    except Exception:
        return False


def screen_stocks(universe):
    """Returns list of dicts for tickers meeting the down-5%+ stock screen."""
    candidates = []

    for symbol in universe:
        try:
            t = yf.Ticker(symbol)
            info = t.info

            market_cap = info.get("marketCap")
            avg_volume = info.get("averageVolume")
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            open_price = info.get("open") or info.get("regularMarketOpen")

            if not all([market_cap, avg_volume, current_price, open_price]):
                continue

            if market_cap < MIN_MARKET_CAP:
                continue
            if avg_volume < MIN_AVG_VOLUME:
                continue

            pct_down_from_open = (open_price - current_price) / open_price * 100
            if pct_down_from_open < DROP_THRESHOLD_PCT:
                continue

            if has_earnings_soon(t):
                continue

            candidates.append({
                "symbol": symbol,
                "current_price": current_price,
                "open_price": open_price,
                "pct_down": pct_down_from_open,
                "market_cap": market_cap,
                "avg_volume": avg_volume,
                "dividend_yield": get_dividend_yield(t, info),
                "ticker_obj": t,
            })

        except Exception as e:
            print(f"[warn] Skipping {symbol}: {e}")
            continue

    return candidates


# ---------------------------------------------------------------------------
# OPTIONS SCREEN
# ---------------------------------------------------------------------------

def find_put_candidates(stock):
    """
    For a given stock (passed the stock screen), scan its put chains across
    all available expirations <= MAX_DAYS_TO_EXPIRATION and return contracts
    meeting the OI / delta criteria.
    """
    t = stock["ticker_obj"]
    S = stock["current_price"]
    q = stock["dividend_yield"]
    results = []

    try:
        expirations = t.options
    except Exception as e:
        print(f"[warn] No options data for {stock['symbol']}: {e}")
        return results

    today = datetime.now(timezone.utc).date()

    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        days_to_exp = (exp_date - today).days

        if days_to_exp <= 0 or days_to_exp > MAX_DAYS_TO_EXPIRATION:
            continue

        T = days_to_exp / 365.0

        try:
            chain = t.option_chain(exp_str)
            puts = chain.puts
        except Exception as e:
            print(f"[warn] Could not fetch chain for {stock['symbol']} {exp_str}: {e}")
            continue

        for _, row in puts.iterrows():
            oi = row.get("openInterest")
            iv = row.get("impliedVolatility")
            strike = row.get("strike")

            if oi is None or iv is None or strike is None:
                continue
            if oi <= MIN_OPEN_INTEREST:
                continue
            if iv <= 0:
                continue

            delta = bsm_put_delta(S=S, K=strike, T=T, r=RISK_FREE_RATE, sigma=iv, q=q)
            if delta is None or abs(delta) >= MAX_ABS_DELTA:
                continue

            bid = row.get("bid")
            ask = row.get("ask")
            if bid is None or ask is None or bid <= 0 or ask <= 0:
                continue
            premium_mid = (bid + ask) / 2

            capital_per_contract = strike * 100
            contracts = math.floor(CAPITAL_PER_POSITION / capital_per_contract)
            if contracts < 1:
                continue

            total_premium = contracts * premium_mid * 100
            if total_premium < MIN_TOTAL_PREMIUM:
                continue

            results.append({
                "symbol": stock["symbol"],
                "expiration": exp_str,
                "days_to_exp": days_to_exp,
                "strike": strike,
                "open_interest": int(oi),
                "implied_vol": iv,
                "delta": delta,
                "bid": bid,
                "ask": ask,
                "premium_mid": premium_mid,
                "contracts": contracts,
                "total_premium": total_premium,
                "capital_deployed": contracts * capital_per_contract,
            })

    return results


# ---------------------------------------------------------------------------
# EMAIL
# ---------------------------------------------------------------------------

def build_email_body(stock_results):
    if not stock_results:
        return "No put-writing candidates found today matching all criteria."

    lines = [
        "Put-Writing Candidates -- Daily Screen",
        "=" * 45,
        "",
        "Criteria: Stock down 5%+ | Mkt cap $5B+ | Avg vol 1M+ | No earnings in 7 days",
        "Options: OI > 1,000 | Expiration <= 180 days | |Delta| < 0.10",
        f"Sizing: ${CAPITAL_PER_POSITION:,.0f} capital budget if assigned | "
        f"Min total premium ${MIN_TOTAL_PREMIUM:,.0f} | Premium = bid/ask midpoint",
        "",
    ]

    for stock_symbol, puts in stock_results.items():
        if not puts:
            continue
        lines.append(f"\n{stock_symbol}")
        lines.append("-" * len(stock_symbol))
        puts_sorted = sorted(puts, key=lambda p: p["total_premium"], reverse=True)
        for p in puts_sorted[:5]:
            lines.append(
                f"  Exp {p['expiration']} ({p['days_to_exp']}d) | "
                f"Strike ${p['strike']:.2f} | "
                f"Delta {p['delta']:.3f} | "
                f"OI {p['open_interest']} | "
                f"Mid premium ${p['premium_mid']:.2f} | "
                f"Contracts {p['contracts']} | "
                f"Total premium ${p['total_premium']:,.0f} | "
                f"Capital deployed ${p['capital_deployed']:,.0f}"
            )

    return "\n".join(lines)


def send_email(subject, body):
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        print("[error] GMAIL_USER / GMAIL_APP_PASSWORD not set. Skipping email send.")
        print(body)
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = RECIPIENT_EMAIL

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())

    print(f"Email sent to {RECIPIENT_EMAIL}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("Screening stocks...")
    stocks = screen_stocks(TICKER_UNIVERSE)
    print(f"{len(stocks)} stocks passed the stock screen: "
          f"{[s['symbol'] for s in stocks]}")

    stock_results = {}
    for stock in stocks:
        print(f"Scanning options for {stock['symbol']}...")
        puts = find_put_candidates(stock)
        stock_results[stock["symbol"]] = puts

    body = build_email_body(stock_results)
    today_str = datetime.now().strftime("%Y-%m-%d")
    subject = f"Put-Writing Candidates -- {today_str}"

    send_email(subject, body)
    print("\n" + body)


if __name__ == "__main__":
    main()
