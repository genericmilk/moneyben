"""Entry point: the once-a-minute trading loop.

Each cycle:
  1. Pull recent 1m candles + spot price for every product (public API).
  2. Compute compact indicators.
  3. (real mode) sync balances from Coinbase.
  4. Ask the LLM for one decision per product.
  5. Clamp each decision against risk limits and execute.
  6. Persist the portfolio and log a one-line P&L summary.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from datetime import datetime

from .coinbase_client import MarketData, TradeClient
from .config import Config
from .indicators import summarize
from .llm import LLMClient
from .portfolio import Portfolio
from .trader import Trader

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(message)s",  # plain lines — we format everything ourselves for readability
)
log = logging.getLogger("moneyben")

_running = True

# Friendly names so beginners see "Bitcoin", not "BTC-USD".
COIN_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "DOGE": "Dogecoin",
    "ADA": "Cardano", "XRP": "XRP", "LTC": "Litecoin", "AVAX": "Avalanche",
    "LINK": "Chainlink", "MATIC": "Polygon", "DOT": "Polkadot",
}


def coin_name(product: str) -> str:
    base = product.split("-")[0]
    return COIN_NAMES.get(base, base)


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def fmt_price(x: float) -> str:
    # Big coins get 2 decimals; sub-dollar coins get more so they aren't "$0.00".
    return f"${x:,.2f}" if x >= 1 else f"${x:,.6f}"


def fmt_holding(product: str, base: float, value: float) -> str:
    """How much of a coin we hold, e.g. '6.7200 SOL ($350.25)' — or '—' if none."""
    if base <= 0:
        return "—"
    amount = f"{base:,.4f}" if base >= 1 else f"{base:,.6f}"
    return f"{amount} {product.split('-')[0]} ({fmt_money(value)})"


def _stop(signum, _frame):
    global _running
    log.info("\n👋 Stopping after this round. Your money is safe and saved.")
    _running = False


def gather_market(md: MarketData, cfg: Config) -> tuple[dict, dict]:
    """Return (market_summary_for_llm, latest_prices)."""
    market: dict = {}
    prices: dict[str, float] = {}
    for product in cfg.products:
        try:
            candles = md.candles(product, cfg.candle_lookback)
            closes = [c["close"] for c in candles]
            if not closes:
                log.info("  %-10s ⚠️  no price data right now — skipping", coin_name(product))
                continue
            spot = md.price(product) or closes[-1]
            prices[product] = spot
            summary = summarize(closes)
            summary["spot"] = round(spot, 6)
            market[product] = summary
        except Exception:  # network blips shouldn't kill the loop
            log.info("  %-10s ⚠️  couldn't reach the exchange — skipping", coin_name(product))
    return market, prices


def run_cycle(cfg: Config, md: MarketData, llm: LLMClient, pf: Portfolio, trader: Trader) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    log.info("\n──────── %s · checking the market ────────", now)

    market, prices = gather_market(md, cfg)
    if not market:
        log.info("  Couldn't get any prices right now. Trying again next round.")
        return

    # Real mode: trust Coinbase for balances rather than our local ledger.
    if cfg.real_trading and trader.trade_client:
        try:
            balances = trader.trade_client.balances()
            pf.sync_from_balances(balances, prices, cfg.products)
        except Exception:
            log.info("  Couldn't check your exchange balance. Doing nothing this round (your money is untouched).")
            return

    snapshot = pf.snapshot(prices)
    try:
        decisions = llm.decide(market, snapshot)
    except Exception:
        log.info("  Couldn't reach the decision helper. Doing nothing this round (your money is untouched).")
        return

    by_product = {d.product: d for d in decisions}
    for product in cfg.products:
        if product not in prices:
            continue
        decision = by_product.get(product)
        if decision is None:
            outcome = "waiting — no suggestion this round"
        else:
            outcome = trader.execute(decision, prices[product])
        holding = fmt_holding(product, pf.base_held(product), pf.position_value(product, prices[product]))
        log.info("  %-10s %-12s  holding %-24s → %s",
                 coin_name(product), fmt_price(prices[product]), holding, outcome)

    pf.save()
    final = pf.snapshot(prices)
    total = final["total_value_usd"]
    pf.ensure_baseline(total)

    # Plain-English profit/loss since the bot started.
    change = total - (pf.start_value or total)
    pct = (change / pf.start_value * 100) if pf.start_value else 0.0
    if abs(change) < 0.01:
        pnl = "no change yet"
    elif change > 0:
        pnl = f"📈 up {fmt_money(change)} ({pct:+.2f}%)"
    else:
        pnl = f"📉 down {fmt_money(abs(change))} ({pct:+.2f}%)"

    coins_value = total - final["cash_usd"]
    log.info("  💰 Your money: %s total  =  %s cash  +  %s in coins   ·   %s",
             fmt_money(total), fmt_money(final["cash_usd"]), fmt_money(coins_value), pnl)


def main() -> int:
    cfg = Config()
    errors = cfg.validate()
    if errors:
        for e in errors:
            log.error("CONFIG: %s", e)
        return 1

    coins = ", ".join(coin_name(p) for p in cfg.products)
    if cfg.real_trading:
        mode_line = "LIVE TRADING — using REAL money! ⚠️"
        money_line = "Trading with the real balance in your Coinbase account"
    else:
        mode_line = "PRACTICE MODE — pretend money, no real trades"
        money_line = f"Starting with {fmt_money(cfg.paper_start_cash)} of pretend money"

    log.info("============================================================")
    log.info("  moneyben — your automatic crypto trader")
    log.info("============================================================")
    log.info("  Mode:          %s", mode_line)
    log.info("  Money:         %s", money_line)
    log.info("  Watching:      %s", coins)
    log.info("  Checks every:  %d seconds", cfg.interval_seconds)
    log.info("  Safety rules:  buys at most %s at a time", fmt_money(cfg.max_trade_usd))
    log.info("                 holds at most %s of any one coin", fmt_money(cfg.max_position_usd))
    log.info("                 always keeps at least %s in cash", fmt_money(cfg.min_cash_reserve_usd))
    log.info("                 only trades when it's reasonably confident")
    log.info("============================================================")

    md = MarketData()
    llm = LLMClient(cfg.openrouter_api_key, cfg.openrouter_model, cfg.openrouter_base_url)
    pf = Portfolio(os.path.join(cfg.data_dir, "portfolio.json"), cfg.paper_start_cash, cfg.quote_currency)

    trade_client = None
    if cfg.real_trading:
        trade_client = TradeClient(cfg.coinbase_api_key, cfg.coinbase_api_secret)
    trader = Trader(cfg, pf, trade_client)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while _running:
        cycle_start = time.time()
        try:
            run_cycle(cfg, md, llm, pf, trader)
        except Exception as e:
            log.exception("Unexpected error in cycle: %s", e)
        # Sleep the remainder of the interval, interruptibly.
        elapsed = time.time() - cycle_start
        to_sleep = max(0.0, cfg.interval_seconds - elapsed)
        slept = 0.0
        while _running and slept < to_sleep:
            time.sleep(min(1.0, to_sleep - slept))
            slept += 1.0

    return 0


if __name__ == "__main__":
    sys.exit(main())
