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
