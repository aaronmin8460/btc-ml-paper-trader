# Strategy Research V3 Notes

This project remains paper-trading only, BTC/USD only, long only, no margin, no leverage, no short selling, no live trading, no fallback trading, and no multi-symbol trading.

## Buy-The-Dip Mean Reversion V2 Rejected

`buy_the_dip_mean_reversion` v2 was evaluated on real BTC/USD `collected_market_data` after the historical backfill reached roughly 180 days of data. The result was valid research data, not synthetic data.

The strategy generated enough trades, so the failure was no longer signal scarcity. In the verified run, 4,927 configs had at least 20 trades and 4,547 configs had at least 50 trades.

The problem was execution economics. Some configs could show positive gross return before costs, but net return failed after conservative fee, slippage, and spread assumptions. Profitable net configs, economically viable configs, and paper-forward eligible configs were all zero.

Conclusion: Buy-the-Dip v2 is a historical rejected strategy. Do not train on it, promote it, connect it to paper-forward, or enable trading from it.

## V3 Research Direction

Strategy Research v3 focuses on higher-timeframe long-only BTC/USD candidates that have a more realistic chance of clearing conservative costs:

- `uptrend_pullback`: buys pullbacks only when BTC is already in a larger uptrend.
- `volatility_breakout`: buys confirmed momentum continuation after a breakout.

The v3 framework supports `15Min` and `1H` research. If raw `1H` bars are unavailable, the research script derives complete hourly bars chronologically from real `15Min` `collected_market_data` and drops incomplete hours. Research decisions remain invalid if synthetic data is used or if the source is not real collected data.

Walk-forward validation is chronological only. A config that works in one segment but fails across the rest is rejected.

## Strategy Reality Audit V5

Reality audit mode is the next research step. It does not add trading permission. It audits why existing configs lose, compares every result against buy-and-hold and DCA baselines, runs current/maker/low/zero-cost scenarios in one pass, and can export trade-by-trade CSV/JSONL logs.

Example:

```bash
.venv/bin/python scripts/research_higher_timeframe.py \
  --audit-mode reality \
  --strategy all \
  --timeframes 1H 4H 1D \
  --max-rows-1h 4000 \
  --max-rows-4h 1200 \
  --max-rows-1d 300 \
  --max-v3-configs 1000 \
  --walk-forward-splits 4 \
  --export-trades \
  --json
```

Reality audit outputs include:

- `logs/higher_timeframe_research.csv`
- `logs/higher_timeframe_research_summary.json`
- `logs/strategy_reality_audit_summary.json`
- `logs/trade_audits/*.csv`
- `logs/trade_audits/*.jsonl`

The current 15Min-heavy families remain historical research strategies and are treated as rejected unless a future reality audit proves otherwise under the full gates: enough trades, positive net return, acceptable profit factor/drawdown, chronological walk-forward pass, and at least one relevant baseline beaten on a risk-adjusted basis.

Rejected-by-default 15Min families:

- 15Min Buy-the-Dip / Mean Reversion v2
- 15Min Trend Pullback
- 15Min Uptrend Pullback if it remains below minimum trade threshold or fails walk-forward
- 15Min Volatility Breakout if it remains net negative

The added higher-timeframe templates are still long-only BTC/USD research only:

- `htf_trend_continuation`
- `htf_volatility_expansion_breakout`
- `htf_risk_off_hold_filter`

`htf_risk_off_hold_filter` is an exposure/regime filter, not a trainable entry strategy.
