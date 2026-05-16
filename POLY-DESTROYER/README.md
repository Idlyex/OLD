# Solana Shares Trader v2

High-performance prediction market trading system. Trades **UP/DOWN shares** on Polymarket-style markets for 5m/15m Solana price intervals. Uses CEX data (Binance, Bybit) as features + Polymarket-specific signals for entry/exit.

## Architecture

```
solana_shares_trader_v2/
├── data/                          # DATA PIPELINE
│   ├── raw/                       # CEX klines + Polymarket market history
│   ├── processed/                 # Merged parquet with all features
│   ├── recorded/shares/           # REAL recorded Polymarket data (by date/duration)
│   ├── collector.py               # Async Binance + Bybit downloader
│   ├── polymarket_collector.py    # Gamma API + CLOB: markets, orderbooks, history
│   └── recorder.py               # Real-time Polymarket recorder (NEW)
├── training/                      # TRAINING PIPELINE
│   ├── train.py                   # LightGBM + CatBoost + meta-stacking
│   ├── walk_forward.py            # Walk-forward rolling window optimization
│   ├── hyper_tuning.py            # Optuna hyperparameter search
│   ├── dataset.py                 # CEX + shares features, shares targets, PurgedKFold
│   └── model_registry/            # Timestamped model snapshots
├── core/
│   ├── features/                  # 104 features across 7 blocks
│   │   ├── price_volume.py        # Block 1: 18 price/volume/VWAP features
│   │   ├── technical.py           # Block 2: 14 ICT/fractal/Wyckoff features
│   │   ├── microstructure.py      # Block 3: 22 order flow features
│   │   ├── liquidation_funding.py # Block 4: 8 liquidation/funding features
│   │   ├── regime.py              # Block 6: 10 HMM/entropy/Hurst features
│   │   ├── shares.py              # Block 7: 16 shares-market features (NEW)
│   │   └── engine.py              # Master feature orchestrator
│   ├── models/                    # Hybrid model architecture
│   ├── risk/                      # Dynamic stops, Kelly sizing
│   ├── execution/                 # Trade lifecycle
│   └── utils/                     # Logger, math helpers
├── strategies/
│   ├── base.py                    # MLHybrid, Microstructure, RegimeAware
│   └── shares.py                  # SharesMispricing, SharesMomentum, SharesHybrid (NEW)
├── backtester/
│   ├── engine.py                  # Legacy CEX backtester
│   ├── shares_engine.py           # Synthetic shares backtester (testing)
│   └── replay_engine.py           # REAL data replay backtester (NEW)
├── live_trader/
│   ├── trader.py                  # Legacy live trader
│   └── shares_trader.py           # Polymarket shares live trader (NEW)
├── dashboard/                     # Rich live console dashboard
├── config/settings.yaml           # Master configuration (incl. shares section)
└── main.py                        # Single entry point (7 modes)
```

## Quick Start — Honest Backtesting

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start recording REAL Polymarket data (run 24h+)
python main.py --mode record --interval 3 --duration 24h
python main.py --mode record --interval 5 --duration infinite   # Ctrl+C to stop

# 3. Check recorded data
python main.py --mode show-recorded

# 4. Replay backtest on REAL data
python main.py --mode backtest --replay --market-duration 15
python main.py --mode backtest --replay --market-duration 5 --strategy shares_mispricing
python main.py --mode backtest --replay --date 2025-05-03 --market-duration 15

# 5. Live trading — dry run (also records data in background)
python main.py --mode live

# 6. Live trading — REAL orders on Polymarket
python main.py --mode live --live
```

## Recorder Details

Records every N seconds from **all active SOL markets** (5m, 15m, 60m):
- **CLOB orderbooks** — bestBid, bestAsk, midPrice, spread, depth (UP + DOWN tokens)
- **Binance SOL/USDT** — bid, ask, mid price
- **Gamma API** — PriceToBeat, market state, accepting_orders
- **Computed features** — momentum_30s/2m, acceleration, liquidity_score, volume_spike, volume_imbalance

Data saved to: `data/recorded/shares/YYYY-MM-DD/{5m,15m,60m}/snapshots.parquet`

Rich live console shows progress bars for each market lifecycle.

## Replay Backtester

Replays recorded data tick-by-tick:
- **Real entry prices** — buy at ask (not mid)
- **Real exit prices** — sell at bid (not mid)
- **Real slippage** — spread-based, from actual orderbooks
- **Real expiry resolution** — SOL price vs PriceToBeat

## Legacy Workflows (still supported)

```bash
# Synthetic backtest (testing/development only)
python main.py --mode backtest --shares --market-duration 15

# CEX-only training
python main.py --mode train --shares --market-duration 15
python main.py --mode train --tune

# Historical market download
python main.py --mode download-markets --days 30
```

## Full Workflow

```
record (24h+) → replay backtest → tune strategy → live
```

| Step | Command | Output |
|------|---------|--------|
| Record | `--mode record --interval 3 --duration 24h` | `data/recorded/shares/DATE/{5,15,60}m/` |
| Show Data | `--mode show-recorded` | Summary of recorded data |
| Replay Backtest | `--mode backtest --replay` | `results/replay_trades.parquet` |
| Train Shares | `--mode train --shares` | `training/model_registry/latest/*.pkl` |
| Live | `--mode live` | Real-time Polymarket shares trading |

## Features (104 total)

| Block | Count | Description |
|-------|-------|-------------|
| Price & Volume | 18 | Multi-TF returns, Garman-Klass/Parkinson/RS vol, VWAP, volume profile |
| Technical | 14 | EMA ribbon, SuperTrend, ICT (FVG/OB/BOS/CHOCH), Wyckoff, fractal dimension |
| Microstructure | 22 | Bid-ask imbalance, CVD, VPIN, flow toxicity, spoofing/iceberg detection |
| Liquidation | 8 | Liq heat, funding rate acceleration, OI proxy, CVD-funding divergence |
| On-Chain | 10 | Whale transfers, DEX volume, Jupiter swaps, MEV bundles, priority fees |
| Regime | 10 | HMM states, Hurst exponent, Shannon/ApEn entropy, autocorrelation decay |
| **Shares** | **16** | **time-to-expiry, distance-from-PTB, mispricing, momentum, volume imbalance, liquidity, spread, mean-reversion, arbitrage** |

## Shares-Specific Features (Block 7)

- `time_remaining_pct`, `time_remaining_min`, `time_elapsed_min`, `life_phase`
- `distance_from_ptb_pct`, `distance_from_ptb_norm` (vol-adjusted)
- `up_implied_prob`, `mispricing_score`
- `shares_momentum_30s/1m/3m`
- `volume_imbalance`, `liquidity_score`, `spread_normalized`
- `mean_reversion_strength`, `arbitrage_score`

## Shares Strategies

- **SharesMispricing**: Buys shares when market price diverges from model probability (edge detection)
- **SharesMomentum**: Follows recent shares price momentum in favorable direction
- **SharesHybrid**: Weighted combination of ML model + mispricing + momentum signals

## Model Architecture

1. **Tabular Head**: LightGBM + CatBoost ensemble (stacking via LogisticRegression)
2. **Multi-task targets**: SOL direction, shares PnL, early exit probability, optimal exit time
3. **Top features**: `distance_from_ptb_norm`, `entropy_approximate`, `autocorr_lag5`, `vol_rogers_satchell`

## Shares Backtester

Simulates realistic prediction market lifecycle:
- New market every N minutes with PriceToBeat = SOL price at open
- Synthesizes UP/DOWN shares prices using Black-Scholes + market noise
- Entry: buy shares when strategy signals (mispricing, momentum)
- Exit: early sell, stop loss, trailing stop, or hold to expiry (shares → $0 or $1)
- Slippage model based on shares liquidity and spread

## Risk Management

- **Hard Stop**: -15% on shares position
- **Trailing Stop**: activates at 30% peak PnL, exits at 60% of peak
- **Dead Share Exit**: if shares price drops below $0.02
- **Expiry Lock**: takes profit when PnL > 10% and < 30s left
- **Max Position Size**: configurable (default $2 per trade)
- **Max Open Positions**: configurable (default 3)
