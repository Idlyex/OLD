# Requirements: NEUROPM

> Формальные требования, выведенные из утверждённого design-документа `d:\AFF\NEUROPM\.kiro\specs\neuropm\design.md`.
> Каждый Acceptance Criterion в этом файле трассируется к конкретной секции design (§), correctness-property (P1..P12 из §20) и/или legacy-багу (L1..L8 из §0/§22).
> Стиль формулировок: EARS (Easy Approach to Requirements Syntax). Модальный глагол — `SHALL`. Никаких «should», «may», «would be nice».

---

## Introduction

NEUROPM — это полный rewrite live-трейдера `solana_shares_trader_v2` в high-performance ML-driven Polymarket shares trader на симбиозе Rust (hot-path: WS, orderbook, feature engine) и Python (orchestrator, business logic, ML inference, recording, dashboard). Источник истины live-поведения — `d:\AFF\AAA\POLY-DESTROYER\` (84% WR, EV $+1.86, gm0.03 sweet spot, 31 сделка по live-данным). Целевой workspace — `d:\AFF\NEUROPM\`. Single OS-process, single Python `asyncio` loop, dedicated `tokio` runtime внутри Rust crate `neuropm-rs` через PyO3, без `multiprocessing`.

Документ требований формализует **что** система должна делать и какие инварианты должны соблюдаться; **как** это устроено — описано в design.md. Каждое требование ниже самостоятельно тестируемо: либо unit-тестом, либо property-тестом (Hypothesis / `proptest`), либо integration-тестом с replay engine, либо measurement-тестом в случае latency-бюджетов. Любое требование, которое нельзя протестировать одним предложением, переписано или удалено.

---

## Glossary

| Term | Definition |
|---|---|
| **PTB** (Price-to-Beat) | Цена базового актива от Pyth/Chainlink на момент открытия рынка; market резолвится UP если `final_price ≥ PTB`. |
| **YES / UP** | Токен, выплачивающий $1 если `final_price ≥ PTB`. |
| **NO / DOWN** | Токен, выплачивающий $1 если `final_price < PTB`. |
| **CLOB** | Central Limit Order Book — exchange-протокол Polymarket. |
| **GTC** | Good-Till-Cancelled order — стоит в книге до match'а или отмены. |
| **FAK** | Fill-And-Kill — исполнить немедленно на любой глубине, остаток отменить. |
| **VWAP** | Volume-Weighted Average Price — средневзвешенная по объёму цена исполнения. |
| **EIP-712 v2** | Стандарт типизированной подписи Ethereum, используемый py-clob-client v2 для подписи ордеров Polymarket. |
| **neg_risk** | Тип PM-маркета на отдельном exchange-контракте; YES+NO суммируются ровно в 1.0 без spread'а. |
| **Pre-signed order** | EIP-712 v2 подписанный payload ордера, удерживаемый в памяти готовым к мгновенному POST. |
| **Auto-takeprofit @0.99** | GTC SELL, выставляемый сразу после BUY fill по цене 0.99 USD за share, исключающий on-chain redeem. |
| **Auto-redeem** | Цикл вызова смарт-контракта `CTFExchange.redeemPositions` каждые 120 s для конвертации resolved-позиций в USDC. |
| **Asset** | First-class типизированная сущность (`SOL / BTC / ETH / HYPE`), проходящая через все слои; никогда не stringly-typed-параметр. |
| **AssetSpec** | Структура (Rust + Python mirror), описывающая asset: pyth_price_id, binance_symbol, pm_market_slugs, durations_min, model_registry_path, feature_spec_version. |
| **EV** | Expected Value per trade в USD — `mean(pnl_usd)`. |
| **WR** | Win Rate — доля сделок с положительным realized PnL. |
| **gm0.03** | Gap-momentum threshold = 0.03% — минимальный directional Pyth move относительно PTB, разрешающий entry. |
| **streak_required** | Минимум подряд идущих confident predictions одного направления перед entry. |
| **streak_interval_s** | Максимальный интервал между predictions внутри streak; превышение сбрасывает streak. |
| **REPRESIGN_TICKS** | Порог изменения best_ask (в тиках 0.01) для инвалидации pre-signed ордера. |
| **Critical / High / Normal / Low** | Приоритеты потоков recording mpsc; см. §17.2. |
| **book.is_ready(token_id)** | Предикат: `last_apply_ts < 5s ago AND seq_no > 0 AND not crossed AND initial snapshot received`. |
| **Healthy / Degraded / Dead** | Три состояния каждого внешнего компонента; см. §10. |
| **FeatureVector** | `#[repr(C)]` slab из 80 `f32` (72 named features + 8 padding); версионируется через `feature_spec_version`. |
| **schema_version** | u16-номер схемы Parquet/JSONL потока; bump при breaking change. |
| **maturin develop** | Команда сборки PyO3-модуля `neuropm_rs.pyd` в `python/neuropm/_native/`. |

---

## Requirements

### Requirement 1: Корректность ценообразования и PnL

**User Story:** As a trader, I want pricing and PnL to be computed strictly from the side of the book that I would actually trade against, so that backtest, dry-run, and live results are not inflated by synthetic mid-prices and stop-loss / trailing logic operates on real exit-value.

#### Acceptance Criteria

1. THE `effective_entry_price` function SHALL compute entry price as VWAP across consumed ASK levels for `order_size_usd`, with no exceptions, in all execution modes (live, dry-run, replay-backtest).
2. WHERE the ASK side has at least one non-empty level, THE `effective_entry_price` function SHALL return `fill_price ≥ best_ask` AND `fill_price ≤ worst_consumed_price`.
3. WHEN `total_ask_dollar_value < order_size_usd`, THEN THE `effective_entry_price` function SHALL flag `PARTIAL_FILL_RISK = true` AND return `shares_filled = sum(s_i across consumed levels)`.
4. THE `mark_to_market` function SHALL use `best_bid` of the held side as the canonical mark price, with no exceptions, in all execution modes (live, dry-run, replay-backtest).
5. THE `mark_to_market` function SHALL NOT use mid-price, ask-price, or any synthetic average as a valuation source for any open position.
6. WHEN a position is closed via auto-takeprofit fill at 0.99, THEN THE realized-PnL accounting SHALL record `exit_price = 0.99` per share filled and SHALL exclude the unfilled remainder from the realized component until that remainder resolves.
7. WHEN a partial auto-takeprofit fill occurs (some shares filled at 0.99, remainder open at resolution), THEN THE total PnL SHALL equal `realized(filled_shares × (0.99 − entry_price)) + at_resolution(remainder_shares × (1.0_or_0 − entry_price)) − fees`.
8. FOR ALL non-empty ASK sequences and sizes `s₁ ≤ s₂`, THE `effective_entry_price` function SHALL satisfy `effective_entry_price(asks, s₁) ≤ effective_entry_price(asks, s₂)` (monotonicity in size).
9. WHEN a fills array `{f_i}` resulting from an order posted with `max_price = P` is received, THEN THE executor SHALL ensure `max_i(f_i.price) ≤ P + slippage_tolerance` and SHALL reject the fill batch and alert otherwise.
10. FOR ALL recorded book snapshots, THE arbitrage-residual `|best_bid_YES + best_ask_NO − 1|` SHALL be bounded by `spread_YES + spread_NO + 0.02`.

#### Traceability
- Design: §14.1, §14.2, §14.3, §14.4, §14.5, §14.6
- Properties: P1, P2, P3, P9, P12
- Legacy fixes: L1

---

### Requirement 2: Pyth Hermes Oracle Integration

**User Story:** As a trader, I want oracle price information delivered through streaming WebSockets / SSE with sub-second freshness, so that gap-momentum and oracle-distance features reflect the current market state and not 1–3 s stale snapshots.

#### Acceptance Criteria

1. THE `neuropm-pyth` crate SHALL consume Pyth Hermes price updates exclusively over the SSE endpoint `/v2/updates/price/stream` with `binary=true`, in all execution modes.
2. THE `neuropm-pyth` crate SHALL NOT use REST polling of Pyth Hermes in the hot-path under any circumstances.
3. WHEN a Pyth SSE binary chunk arrives, THEN THE VAA decoder SHALL parse it into a `PriceTick {asset, ts_recv_ns, ts_oracle_ms, price, conf_pct}` within p99 ≤ 150 µs, measured per-tick over a 1-minute window of ≥ 100 ticks.
4. WHEN the inter-tick gap from Pyth Hermes exceeds 5 s, THEN THE oracle health monitor SHALL transition the Pyth stream to `Dead` state, halt all entries, mark all open positions as `stale-oracle`, switch mark-to-market sourcing to bid-only-from-book, and emit a Telegram alert.
5. WHILE the Pyth stream is in `Degraded` state (gap 1–5 s), THE strategy SHALL continue to evaluate signals but SHALL annotate every emitted `Prediction` with `oracle_health = degraded` for downstream gating.
6. IF `|pyth_price − binance_mark_price| / pyth_price > 0.005` is sustained for more than 10 s, THEN THE oracle-skew circuit breaker SHALL halt all entries and flag potential oracle attack.
7. FOR ALL resolved markets in `outcomes.jsonl`, the recorded outcome SHALL be consistent with the Pyth price at `end_ts` relative to the recorded `PTB`: `pyth_at_end ≥ PTB ⇔ outcome = UP`.
8. WHERE a Pyth Hermes connection drops, THE `neuropm-pyth` crate SHALL reconnect with exponential backoff (1 s, 2 s, 4 s, …, 30 s cap) and SHALL refetch the latest price for each subscribed `pyth_price_id` before resuming feature emission.

#### Traceability
- Design: §1 goal 4, §3.1, §3.2 (S0, S1), §10 (Pyth Hermes row), §10.1 (oracle-skew circuit), §11.6, §15.1 (oracle-derived features), §15.5 (gap_momentum_check)
- Properties: P5
- Legacy fixes: L2

---

### Requirement 3: Polymarket CLOB Integration & Orderbook Maintenance

**User Story:** As a trader, I want a defensively-correct L2 book that survives PM event quirks (unsorted payload, seq gaps, duplicates, crossed states), so that downstream features and pricing are never computed on a malformed book.

#### Acceptance Criteria

1. THE PM CLOB market WS client SHALL stream events exclusively from `wss://ws-subscriptions-clob.polymarket.com/ws/market`, with no REST polling of `/book` in the hot-path.
2. WHEN a `book` event is received, THEN THE `apply_book_event` algorithm SHALL replace state, sort BIDS descending and ASKS ascending, and update `seq_no` and `last_apply_ts`.
3. WHEN a `price_change` event is received with `seq_no = current_seq_no`, THEN THE `apply_book_event` algorithm SHALL treat it as a duplicate and SHALL NOT mutate the book.
4. WHEN a `price_change` event is received with `seq_no > current_seq_no + 1`, THEN THE `apply_book_event` algorithm SHALL return `ResyncRequired`, transition the book to `NotReady`, fetch a full REST snapshot via `GET /book?token_id=…`, drain the WS replay buffer for events with `seq_no > snapshot_seq`, and re-arm the book.
5. WHEN `price_change` deltas include a level with `size = 0`, THEN THE `upsert_or_remove` operation SHALL remove that price level rather than insert a zero-size level.
6. AFTER every applied book event, THE `apply_book_event` algorithm SHALL satisfy `bids[0].price < asks[0].price` OR at least one side empty (no-crossed-book invariant).
7. IF two consecutive crossed-book states are detected, THEN THE book SHALL be force-resynced via REST snapshot before any further inference is gated to `Ready`.
8. WHILE `book.is_ready(token_id) = false`, THE strategy SHALL NOT call `predict_and_route` for that token, and THE executor SHALL NOT post any new orders for that token.
9. THE `parse_levels` routine SHALL sort the input payload before any `book[0]` / `levels[0]` access, in every code path (live, replay, fixture-load), so that PM's unsorted-payload behaviour cannot mislead `best_bid` / `best_ask` computation.

#### Traceability
- Design: §11.4, §13.1, §13.2, §13.3 (defensive rules), §13.4 (resync algorithm), §10 (PM CLOB market WS row), §10.2 (reconnect protocol)
- Properties: P10, P11
- Legacy fixes: L5

---

### Requirement 4: ML Inference Pipeline

**User Story:** As a trader, I want event-driven 200 ms ML inference that uses ONNX Runtime exclusively and applies streak + gap-momentum + transformer-veto filters before signal emission, so that entries are both more frequent than legacy 5 s polling and more disciplined than raw model output.

#### Acceptance Criteria

1. THE `ml/runtime.py` module SHALL load and serve all four ensemble members (`cb`, `lgbm`, `xgb`, `rf`) and the optional `transformer` exclusively as ONNX sessions, with no `joblib` / `pickle` model deserialization permitted in the live trader.
2. AT process startup, THE `ml/runtime.py` module SHALL execute at least 50 dummy inferences per loaded ONNX session before the orchestrator marks the asset as `inference_ready = true`.
3. THE `tick_pump` task SHALL trigger inference on `book_event` callbacks debounced to a minimum interval of 200 ms per asset, with multiple events within the window coalesced to exactly one inference call.
4. WHILE no `book_event` arrives for ≥ 1 s for an active asset, THE `tick_pump` task SHALL still trigger one heartbeat inference call so that oracle-driven feature changes are not stalled.
5. WHEN an emitted `FeatureVector` contains any non-finite value (`NaN` or `±Inf`), THEN THE inference task SHALL skip that tick, increment `metrics.feat_nan_count`, and record the skip into `predictions.parquet` with `decision = "skip_nan"`.
6. THE `feature_emit` routine in `neuropm-feats` SHALL use only data with `ts ≤ t` when emitting a `FeatureVector` at time `t`, with no future-leakage permitted.
7. FOR EVERY active asset, THE `FeatureVector` SHALL contain exactly 80 `f32` slots — 72 named features (16 shares + 36 microstructure + 12 oracle + 8 time/context) and 8 zero-padding slots — pinned by `feature_spec_version`.
8. WHEN `feature_spec_version` in a loaded model bundle does not equal the version compiled into the running `neuropm-feats` crate, THEN THE orchestrator SHALL fail fast at startup with an explicit version-mismatch error and SHALL NOT enter live mode.
9. WHEN model raw probabilities are computed, THEN THE pipeline SHALL apply per-model isotonic / Platt calibration from `calibration.json` before any comparison against `min_confidence` or before ensemble averaging.
10. WHERE `transformer_enabled = true` for an asset AND `sign(p_transformer − 0.5) ≠ sign(p_ensemble − 0.5)` AND `|p_transformer − 0.5| > VETO_MARGIN`, THE strategy SHALL veto the entry and record `decision = "veto"` in `predictions.parquet`.
11. WHEN evaluating streak entry-readiness, THE `streak_check` filter SHALL require `streak_required` consecutive predictions of identical direction, each with calibrated confidence ≥ `min_confidence`, with maximum inter-prediction gap ≤ `streak_interval_s`.
12. WHEN evaluating gap-momentum, THE `gap_momentum_check` filter SHALL accept the entry only if `direction_signed_gap_pct ≥ gap_min_pct` where `direction_signed_gap_pct = ((sol_price − ptb) / ptb × 100) × (+1 if UP else −1)`.

#### Traceability
- Design: §3.2 (S6, S7), §15.1 (feature composition), §15.2 (models, ONNX-only), §15.3 (200 ms cadence), §15.4 (streak filter), §15.5 (gap-momentum), §15.6 (calibration), §15.7 (ensemble), §15.8 (ONNX Runtime), §15.9 (inference loop)
- Properties: P4
- Legacy fixes: L3, L4

---

### Requirement 5: Pre-signing & Order Execution State Machine

**User Story:** As a trader, I want EIP-712 v2 orders pre-signed against the current best_ask so that on signal arrival the in-process latency between intent and POST is minimised, and I want a deterministic state machine governing the lifecycle Idle → PresignedReady → Posting → Filled → SellArming → Holding → {SellHit | Resolved | EmergencyExit}.

#### Acceptance Criteria

1. THE `presigner_task` SHALL maintain at most one valid pre-signed BUY order per `(token_id, direction)` pair, pinned to the current `best_ask` and the configured `order_size_usd`.
2. WHEN `|best_ask − pinned_to_ask| ≥ REPRESIGN_TICKS × tick_size`, THEN THE pre-signed order SHALL be invalidated and a new pre-sign SHALL be initiated before any new entry posts for that token.
3. WHEN `now − presigned.created_ts > 30 s`, THEN THE pre-signed order SHALL be force-refreshed regardless of price drift.
4. WHILE `best_ask > asset_spec.entry.max_share_price` OR `best_ask < asset_spec.entry.min_share_price`, THE `presigner_task` SHALL hold `presigned[token_id] = None` and SHALL NOT consume signing CPU.
5. WHEN a BUY signal is dispatched and a valid pre-signed order exists, THEN THE executor SHALL POST the pre-signed bytes directly to `clob.polymarket.com/order` without re-signing, in p99 wall-clock ≤ 150 ms (network-bounded).
6. WHEN a BUY signal is dispatched and no valid pre-signed order exists, THEN THE executor SHALL fall back to sign-then-post and SHALL increment `metrics.presign_miss_total`.
7. WHEN a BUY fill is reported via PM user WS (`order_matched`, full or partial), THEN THE executor SHALL place a GTC SELL @ 0.99 for the filled share quantity, with the correct `neg_risk` flag matching the market.
8. WHEN a placed order has not received a fill within 10 s and is no longer top-of-book, THEN THE executor SHALL transition to `Cancelling` state and issue a cancel request.
9. WHILE the orchestrator is performing graceful shutdown, THE executor SHALL cancel all open BUY orders, SHALL NOT cancel any GTC SELL @ 0.99 orders, and SHALL allow the recorder up to 30 s of drain time before process exit.
10. THE `neg_risk` flag passed to every `clob_client.post_order` call SHALL be derived from the resolved market metadata via `_resolve_neg_risk` and SHALL NOT be hardcoded.

#### Traceability
- Design: §16.1 (state machine), §16.2 (pre-sign details), §16.3 (auto-takeprofit), §16.5 (cancel-on-shutdown), §14.5 (neg_risk), §3.2 (S9, S10)
- Properties: P3, P8

---

### Requirement 6: Recording Pipeline

**User Story:** As an operator and as a future-model trainer, I want every market event, prediction, order, fill, and outcome recorded losslessly to local Parquet/JSONL with deterministic backpressure semantics, so that postmortem analysis and future training datasets are complete.

#### Acceptance Criteria

1. THE recording pipeline SHALL fan out producer events through a single `asyncio.Queue` of capacity 4096 to a dedicated `recorder_writer_task` that drains and batches per stream.
2. THE recorder SHALL write all Parquet I/O through `asyncio.to_thread` and SHALL NOT execute synchronous Parquet writes from the main event loop.
3. FOR ALL events with `priority = Critical` (predictions, orders, fills, outcomes), THE enqueue routine SHALL use blocking put semantics, SHALL NOT drop, and EVERY such event SHALL appear as a record in the corresponding stream's Parquet/JSONL file with matching `ts` and `payload`.
4. WHEN queue utilization exceeds 80%, THEN THE enqueue routine SHALL drop the oldest `Normal`-priority events (book_diffs) before dropping any newly-arrived `Normal` events.
5. WHEN queue utilization exceeds 95%, THEN THE enqueue routine SHALL drop the oldest `High`-priority events (oracle_ticks) only after exhausting `Normal` and `Low` queues, and SHALL NEVER drop `Critical` events.
6. THE recorder SHALL increment `metrics.{normal,high,low}_dropped_total` counters on every drop and SHALL emit a Telegram warning when any non-Critical drop counter increments by ≥ 1000 events within a 60 s window.
7. AT UTC midnight, THE recorder SHALL flush all open buffers, close current Parquet writers, and rotate output paths to `data/recorded/{asset}/{YYYY-MM-DD}/` for the new day.
8. WHEN any JSONL.zst file reaches 256 MB, THEN THE recorder SHALL roll to a new file segment within the same daily directory.
9. THE `session_meta.json` file SHALL record `session_id`, `git_sha`, `config_hash`, the schema_version for every active stream, `started_at`, and the active `assets` list, written at session start and updated on shutdown.
10. WHEN any Parquet writer is closed, THEN the resulting file SHALL be readable by `pyarrow.parquet.read_table` without error and SHALL contain a `schema_version` column equal to the version recorded in `session_meta.json`.

#### Traceability
- Design: §6.1 (storage architecture), §6.4 (schema versioning), §17.1 (topology), §17.2 (backpressure), §17.3 (schema versions), §17.4 (daily rotation)
- Properties: P6
- Legacy fixes: L7

---

### Requirement 7: Multi-Asset Support & First-Class `Asset` Abstraction

**User Story:** As a developer, I want assets (SOL/BTC/ETH/HYPE) to flow through every layer as a typed `AssetSpec` rather than a stringly-typed parameter, so that adding BTC/ETH/HYPE in Phase 2 requires only a config file and a model registry, not a shotgun edit.

#### Acceptance Criteria

1. THE Rust core SHALL define `Asset` as an `enum { Sol, Btc, Eth, Hype }` and `AssetSpec` as a struct carrying `pyth_price_id: [u8;32]`, `binance_symbol`, `pm_market_slugs`, `durations_min`, `feature_spec_version`.
2. THE Python domain layer SHALL expose `domain/asset.py::AssetSpec` as a `pydantic` model mirroring the Rust struct field-for-field with validated types.
3. THE conversion from string keys (`"SOL"`, `"BTC"`, etc.) to typed `Asset` / `AssetSpec` values SHALL happen exactly once in `config/loader.py` at startup; no string-to-asset coercion SHALL appear in `runtime/`, `strategy/`, `execution/`, `recording/`, or `ml/`.
4. NO function signature in `runtime/`, `strategy/`, `execution/`, `recording/`, or `ml/` SHALL accept a bare `str` describing an asset; assets SHALL be passed exclusively as `Asset` or `AssetSpec`.
5. PER-ASSET state in both Python and Rust SHALL be stored in `dict[Asset, X]` / `HashMap<Asset, X>` containers; no parallel `sol_*` / `btc_*` field naming SHALL appear in production code.
6. WHEN an asset configuration has `enabled = true`, THEN the corresponding `data/model_registry/{asset}/latest/` directory SHALL exist and SHALL contain `ensemble_cb.onnx`, `ensemble_lgbm.onnx`, `ensemble_xgb.onnx`, `ensemble_rf.onnx`, `feature_spec.json`, `calibration.json`, and `meta.json`; otherwise startup SHALL fail fast with an explicit error.
7. WHILE only one asset is currently `enabled = true`, THE recorder SHALL still respect the multi-asset path layout `data/recorded/{asset}/{YYYY-MM-DD}/` so that Phase 1.5 (record-only for additional assets) is a config-flag flip.

#### Traceability
- Design: §1 goal 8, §8.1 (Rust Asset enum), §8.2 (Python mirror), §8.3 (composition without duplication), §8.4 (strategy & models per-asset), §8.5 (phase plan)
- Legacy fixes: L8

---

### Requirement 8: Performance & Latency Budgets

**User Story:** As a trader, I want a measurable, enforceable latency budget per pipeline stage so that the in-process tick lifecycle stays inside its design envelope, and so that regressions trigger circuit breakers instead of silently degrading entries.

#### Acceptance Criteria

1. WHEN end-to-end in-process latency from `ts_recv_ns(book_event)` to `inference_emit_ts` is measured, THE system SHALL achieve p99 ≤ 4.5 ms over a 1-minute window of ≥ 100 ticks, with all measurements emitted via `tracing` spans (Rust) and `time.perf_counter_ns()` (Python) into `StateRegistry`.
2. WHEN Pyth VAA decode latency is measured per chunk, THE system SHALL achieve p99 ≤ 150 µs and hard limit ≤ 500 µs.
3. WHEN PM CLOB WS message parse latency (`recv → simd-json parsed`) is measured per message, THE system SHALL achieve p99 ≤ 250 µs and hard limit ≤ 1 ms.
4. WHEN L2 incremental update latency is measured per applied event, THE system SHALL achieve p99 ≤ 500 µs and hard limit ≤ 2 ms.
5. WHEN feature-vector assembly latency is measured per emission, THE system SHALL achieve p99 ≤ 200 µs and hard limit ≤ 1 ms.
6. WHEN the PyO3 boundary cross + NumPy view construction latency is measured, THE system SHALL achieve p99 ≤ 50 µs and hard limit ≤ 200 µs.
7. WHEN ensemble ONNX inference latency (4 models, batch=1) is measured, THE system SHALL achieve p99 ≤ 1.5 ms and hard limit ≤ 5 ms; ensemble members SHALL run in parallel via `asyncio.gather` over `to_thread`, so combined wall-time ≈ max(individual), not sum.
8. WHEN transformer ONNX inference latency (60 × N seq, batch=1) is measured, THE system SHALL achieve p99 ≤ 1.5 ms and hard limit ≤ 5 ms.
9. WHEN strategy gates latency (streak + gap_momentum + price/spread/depth) is measured, THE system SHALL achieve p99 ≤ 100 µs and hard limit ≤ 500 µs.
10. WHEN executor decision + presigned-lookup latency is measured, THE system SHALL achieve p99 ≤ 100 µs and hard limit ≤ 200 µs.
11. IF a per-stage hard limit is breached, THEN THE observability layer SHALL emit a Telegram warning naming the stage and the breach magnitude.
12. IF a per-stage hard limit is breached three consecutive times for the same stage within 30 s, THEN THE latency circuit breaker SHALL halt all entries until p99 returns inside its hard limit for ≥ 60 s.

#### Traceability
- Design: §3.2 (end-to-end latency table), §9 (per-operation budget table), §10.1 (latency circuit breaker)

---

### Requirement 9: Failure Handling & Circuit Breakers

**User Story:** As an operator, I want every external dependency to have explicit Healthy / Degraded / Dead states with deterministic transitions and recovery procedures, so that partial failures degrade gracefully and never cause silent bad-data trading.

#### Acceptance Criteria

1. WHEN the inter-event gap on PM CLOB market WS exceeds 10 s for any subscribed `token_id`, THEN THE WS supervisor SHALL transition that connection to `Dead`, reconnect with exponential backoff (1 s, 2 s, 4 s, …, 30 s cap), and SHALL NOT resume inference for that token until a full REST `/book` snapshot has been fetched and the WS replay buffer has been drained.
2. WHILE a PM market WS is in `Dead` state for any subscribed token, THE strategy SHALL NOT call `predict_and_route` for that token, and THE executor SHALL NOT post any new orders for that token.
3. WHEN PM user WS misses three consecutive heartbeats, THEN THE WS supervisor SHALL reconnect AND SHALL fall back to REST `/orders` polling at 2 s cadence to track pending fills until the user WS recovers.
4. WHEN the inter-event gap on Binance Futures WS exceeds 30 s, THEN THE feature engine SHALL continue emitting feature vectors using stale CEX features for up to 60 s; AFTER 60 s of staleness, THE ensemble SHALL drop microstructure-heavy models (`xgb`, `rf` per `feature_spec`) from voting until Binance WS recovers.
5. WHEN ONNX inference returns `NaN` or `Inf` for any model in a single tick, THEN THE inference task SHALL skip that tick; AFTER 5 consecutive `NaN`/`Inf` results for the same model, THAT model SHALL be disabled in the ensemble for the remainder of the session and a Telegram alert SHALL be emitted.
6. THE daily-loss circuit breaker SHALL halt all new entries when `cumulative_realized_pnl_today_usd < −daily_loss_limit_usd`, and SHALL leave only exit-side logic (auto-takeprofit, auto-redeem, manual cancel) running.
7. THE consecutive-loss circuit breaker SHALL halt all new entries for 1 hour when `consecutive_losing_trades ≥ kill_switch_consecutive_losses` (default 8).
8. WHEN free disk space in `data_dir` falls below 1 GB, THEN THE recorder SHALL aggressively compress yesterday's directory immediately; WHEN free disk falls below 100 MB, THEN THE recorder SHALL refuse new writes and emit a Telegram alert.
9. WHEN a Polygon RPC call (auto-redeem) times out three consecutive times, THEN THE auto-redeem task SHALL stop retrying for the current 120 s cycle, leave positions un-redeemed, and reattempt at the next cycle; this failure SHALL NOT halt trading.
10. AFTER any WS reconnect is established for a `token_id`, THE inference for that token SHALL remain gated until `book.is_ready(token_id) = true`.

#### Traceability
- Design: §10 (failure model table), §10.1 (circuit breakers), §10.2 (reconnect protocol), §13.4 (resync algorithm)
- Properties: P7

---

### Requirement 10: Risk Management

**User Story:** As a trader, I want hard caps on position size, daily loss, consecutive losses, and per-asset inventory enforced inside the executor, so that no buggy strategy or oracle glitch can drain the account.

#### Acceptance Criteria

1. WHILE `len(open_positions) ≥ max_open_positions`, THE executor SHALL reject all new BUY signals with `reason = "max_open_positions"`.
2. WHEN a BUY signal SUCH THAT `inventory_usd[asset] + signal_size_usd > max_inventory_usd_per_asset` is dispatched, THEN THE executor SHALL reject the signal with `reason = "max_inventory_usd_per_asset"`.
3. THE risk manager SHALL compute realized daily PnL using only filled orders confirmed via PM user WS (or REST fallback per Requirement 9), with no estimation from mark-to-market.
4. WHEN `cumulative_realized_pnl_today_usd ≤ −daily_loss_limit_usd`, THEN THE risk manager SHALL transition to `entries_halted` state and SHALL persist this state across orchestrator restarts within the same UTC day.
5. WHEN `consecutive_losing_trades ≥ kill_switch_consecutive_losses`, THEN THE risk manager SHALL halt new entries for exactly `1 h` from the timestamp of the triggering loss, after which the counter SHALL reset to zero.
6. THE risk manager SHALL refuse to start the orchestrator if `daily_loss_limit_usd ≤ 0`, `max_open_positions ≤ 0`, `max_inventory_usd_per_asset ≤ 0`, or any cap is missing from `trading.toml` or asset overrides.
7. WHEN `entries_halted` is active, THE executor SHALL still execute auto-takeprofit fills, auto-redeem cycles, and manual cancellations, but SHALL NOT post any new BUY orders.
8. WHEN a Telegram `/halt` command is received from the configured operator, THEN THE risk manager SHALL transition immediately to `entries_halted` and SHALL acknowledge the command.

#### Traceability
- Design: §10.1 (circuit breakers), §7.3 (`[risk]` table in `trading.toml`)

---

### Requirement 11: Configuration & Validation

**User Story:** As a developer and as an operator, I want all configuration loaded from typed TOML files validated by `pydantic` v2 at startup, so that misconfigurations fail fast with a clear error and no implicit defaults silently change behaviour.

#### Acceptance Criteria

1. THE configuration loader SHALL accept exclusively TOML files (`settings.toml`, `trading.toml`, `assets/{asset}.toml`) and SHALL NOT accept YAML or JSON for runtime configuration.
2. THE configuration loader SHALL merge configs in the strict precedence order `settings → trading → asset → env-overrides` and SHALL fail fast on any unknown top-level key.
3. THE configuration loader SHALL validate the merged result with `pydantic` v2 models, where every field used by any code path is declared explicitly with either a documented default or marked `required`.
4. WHERE an asset config has `enabled = true`, THE configuration validator SHALL verify that `model.registry_id` resolves to an existing directory under `data/model_registry/` and SHALL fail fast otherwise.
5. THE configuration validator SHALL verify each `pyth_price_id` is exactly 32 bytes (64 hex chars after `0x` prefix) and SHALL reject any malformed value at startup.
6. THE configuration validator SHALL verify `min_share_price < max_share_price` AND `min_entry_pct < max_entry_pct` AND `gap_min_pct ≥ 0` AND `streak_required ≥ 1` AND `streak_interval_s > 0`, and SHALL reject any violating asset config at startup.
7. WHEN startup detects no asset with `enabled = true`, THEN THE orchestrator SHALL refuse to enter live mode and SHALL emit an explicit error specifying which assets were checked and rejected.
8. WHEN running in `dry-run` mode, THE configuration loader SHALL still validate all rules above identically to live mode, so that a `dry-run` rehearsal proves the live config valid.

#### Traceability
- Design: §7.1 (TOML choice), §7.2 (hierarchy), §7.3 (schema), §7.4 (validation rules)

---

### Requirement 12: Observability — Dashboard, Logging, Metrics

**User Story:** As an operator, I want a local dashboard, structured logs, and exposed metrics so that I can see system health, latency histograms, recorder backpressure, ML decisions, and PnL in real time without attaching a debugger.

#### Acceptance Criteria

1. THE dashboard server SHALL bind to `127.0.0.1:8787` only and SHALL NOT bind to any external interface in MVP.
2. THE dashboard server SHALL expose `/api/state` returning an atomic `StateSnapshot` containing per-asset book best-bid/ask, last prediction, open positions, presigned status, latency histogram summaries, and recorder drop counters, all updated at ≤ 1 s lag from real time.
3. THE dashboard server SHALL expose `/api/metrics` returning Prometheus-compatible counters and histograms for every per-stage latency budget defined in Requirement 8.
4. THE logger SHALL write structured JSON log records to `data/logs/neuropm.log` with rotation at 64 MB per file and retention 30 days, configurable via `[logging]` section of `settings.toml`.
5. THE logger SHALL NEVER write `PRIVATE_KEY`, mnemonic, raw EIP-712 signatures, or any secret value at any log level.
6. WHEN any per-stage latency hard limit is breached, THEN THE Telegram bot SHALL emit a warning containing stage name, observed value, hard limit, and a 1-minute trailing p99.
7. WHEN any circuit breaker transitions, THEN THE Telegram bot SHALL emit a status message naming the breaker, the trigger condition, and the recovery condition.
8. THE Telegram bot SHALL accept the operator commands `/status`, `/halt`, `/resume`, `/positions`, and `/pnl` from the configured `chat_id` and SHALL reject all other inbound messages.

#### Traceability
- Design: §2.2 (container diagram dashboard/telegram), §3.2 (where measured), §9 (budget enforcement), §17.2 (drop counters), §21.7 (authenticated dashboard deferred)

---

### Requirement 13: Security & Secrets

**User Story:** As an operator, I want all private keys, RPC URLs, and API tokens read exclusively from `.env`, never logged, and never committed, so that operational security does not depend on developer discipline alone.

#### Acceptance Criteria

1. THE configuration loader SHALL read `PRIVATE_KEY`, `PROXY_WALLET`, `TG_TOKEN`, `TG_CHAT_ID`, and `POLYGON_RPC_URL` exclusively from the process `.env` file (or OS environment), and SHALL NOT accept any of these values in TOML config files.
2. WHEN any secret-bearing variable is missing at startup, THEN THE orchestrator SHALL refuse to enter live mode and SHALL print a redacted error naming the missing variable, with no value echoed.
3. THE `.gitignore` SHALL include `.env`, `target/`, `data/recorded/`, `data/logs/`, `data/model_registry/`, and `__pycache__/`.
4. THE repository SHALL include `.env.example` with every required variable name and a placeholder value, but SHALL NOT include any real `.env`.
5. THE logger and Telegram bot SHALL redact any string matching the loaded `PRIVATE_KEY` value or any 64+ hex-character substring before emission.
6. THE EIP-712 signing path SHALL keep raw signature bytes in process memory only and SHALL NOT persist them to disk except as part of an outgoing order POST body recorded in `orders.jsonl` with the `signature` field redacted to its first 8 hex chars and length suffix.
7. WHEN any HTTP client is constructed, THEN THE request layer SHALL set TLS verification on by default and SHALL fail any request to a non-HTTPS endpoint listed under `[infra.*]`, except for `127.0.0.1` development URLs.

#### Traceability
- Design: §4 (`.env`, `.env.example` in tree), §19.1 (root file purposes)

---

### Requirement 14: Process Topology & Async Model

**User Story:** As a maintainer, I want a single OS process with a single `asyncio` loop and a single dedicated `tokio` runtime, all PyO3-bridged with explicit GIL discipline, so that there is no `multiprocessing`-induced IPC overhead and no uncoordinated task starvation.

#### Acceptance Criteria

1. THE orchestrator SHALL run as a single OS process containing one Python `asyncio` event loop, one Rust `tokio` runtime, and one `concurrent.futures.ThreadPoolExecutor` with 4 worker threads.
2. THE orchestrator SHALL NOT use `multiprocessing`, `os.fork`, or any cross-process state primitive in the MVP execution path.
3. THE Rust hot-path operations (WS recv, simd-json parse, L2 book mutate, feature compute) SHALL release the Python GIL via `Python::allow_threads` for their entire duration.
4. EACH `token_id` book state SHALL be mutated by exactly one `tokio` task (`book_apply_task[token_id]`); reads from feature engine and Python SHALL go through `arc_swap::ArcSwap<OrderBookSnapshot>` lock-free loads.
5. ON graceful shutdown, the orchestrator SHALL cancel tasks in this order: producers (WS, feature emit, tick pump, inference, strategy) → executor with `flush_grace_s = 5` → recorder writer with drain deadline 30 s → process exit; THE recorder writer task SHALL be the last task to terminate.
6. THE clock-drift monitor SHALL emit a warning if `|monotonic_delta − wall_clock_delta|` exceeds 50 ms over any 10 s sample window.
7. ONNX `Run()` calls SHALL be invoked through `asyncio.to_thread` so that the GIL is released for the duration of native inference.
8. NO Python coroutine in the hot-path (tick_pump → inference → strategy → executor) SHALL hold the GIL during a blocking I/O call; HTTP POST of orders SHALL run inside `to_thread`.

#### Traceability
- Design: §5.1 (process topology rationale), §5.2 (task map), §18.1 (runtime split), §18.2 (per-token serialization), §18.3 (GIL strategy)
- Legacy fixes: L6

---

### Requirement 15: Backtest & Replay Engine

**User Story:** As a researcher, I want a deterministic replay engine that consumes recorded WS events and reproduces feature vectors, predictions, and simulated fills bit-for-bit identical to the live session, so that strategy changes and model updates can be evaluated against real market microstructure without paper-trading lag.

#### Acceptance Criteria

1. THE `backtest/replay_engine.py` SHALL consume `book_diffs.jsonl.zst`, `oracle_ticks.parquet`, and `cex_microstructure.parquet` from a specified date range under `data/recorded/{asset}/{YYYY-MM-DD}/` and SHALL produce the same `FeatureVector` sequence as the live session within `f32` epsilon `1e-6` per slot.
2. THE replay engine SHALL drive the same `neuropm-feats` Rust crate used in live mode, SHALL load the same model bundle, and SHALL apply the same calibration and ensemble combiner.
3. THE `backtest/fill_simulator.py` SHALL simulate BUY fills as VWAP across recorded ASK depth at the moment of the simulated POST, using the same `effective_entry_price` algorithm as live execution.
4. THE fill simulator SHALL simulate auto-takeprofit @ 0.99 fills against recorded BID depth and resolution events, producing realized-PnL identical in formula to live PnL accounting (Requirement 1).
5. WHEN replaying a recorded session twice with identical inputs and identical config, THE replay engine SHALL produce identical `predictions.parquet` and identical simulated fills.
6. THE backtest CLI SHALL emit metrics WR, EV, Sharpe, max-drawdown, profit-factor, and trade-count per asset per run, written to `data/results/backtest/{run_id}/metrics.json`.
7. THE backtest engine SHALL NOT perform any network call to PM, Pyth, Binance, or Polygon during replay execution.

#### Traceability
- Design: §19.11 (`backtest/` layout), §14.7 (sources cited; recorded data is replay-grade), §17.1 (`book_diffs` is byte-equivalent)

---

### Requirement 16: Deployment & Tooling (Windows-native)

**User Story:** As a developer on Windows, I want one-shot setup, build, and run scripts that work without WSL, Docker, or POSIX-only assumptions, so that onboarding takes minutes and not days.

#### Acceptance Criteria

1. THE PyO3 module `neuropm_rs.pyd` SHALL be built via `maturin develop` invoked from the workspace root, with output placed in `python/neuropm/_native/`.
2. THE Rust toolchain SHALL be pinned to stable `1.79+` via `rust-toolchain.toml`, and the build SHALL fail fast on any older toolchain.
3. THE Python build SHALL use `pyproject.toml` with `hatchling + maturin` backend, target Python `3.11` ABI3 (`abi3-py311`), and SHALL NOT depend on any Linux-only library at runtime.
4. THE `scripts/dev_setup.ps1` PowerShell script SHALL create a virtualenv, install Python dependencies, run `maturin develop` once, and install pre-commit hooks; ALL steps SHALL succeed on a clean Windows 10/11 dev box.
5. THE `scripts/run_live.ps1` script SHALL launch the orchestrator with `dry-run = true` by default, requiring an explicit `-Live` switch to enter live trading.
6. THE `scripts/run_record_only.ps1` script SHALL launch the orchestrator with execution disabled and recording enabled for all `enabled = true` assets.
7. THE `scripts/run_backtest.ps1` script SHALL invoke `python -m neuropm backtest --asset SOL --from YYYY-MM-DD --to YYYY-MM-DD` and SHALL emit results to `data/results/backtest/{run_id}/`.
8. THE codebase SHALL NOT call `os.fork`, raw `epoll`, raw `inotify`, or any other POSIX-only primitive at any code path.
9. THE Parquet, JSONL, and zstd I/O paths SHALL function on NTFS without requiring case-sensitivity or any non-default filesystem feature.

#### Traceability
- Design: §1 goal 9 (Windows-native), §4 (root file list, scripts), §11.1 (Rust workspace), §19.13 (scripts)

---

## Acceptance for moving to Tasks phase

This requirements document is "done" when:
1. Every legacy bug L1..L8 from design §0/§22 appears as at least one Acceptance Criterion in this file (✅: L1→Req 1.4–1.5, L2→Req 2.2, L3→Req 4.1, L4→Req 4.6 + Req 8.5, L5→Req 3.9, L6→Req 14.1–14.2, L7→Req 6.2, L8→Req 7.3–7.5).
2. Every correctness property P1..P12 from design §20 is referenced in at least one Acceptance Criterion's Traceability subsection (✅: P1/P2/P3/P9/P12→Req 1, P4→Req 4, P5→Req 2, P6→Req 6, P7→Req 9, P8→Req 5, P10/P11→Req 3).
3. Every quantitative latency budget from design §3.2 / §9 has a SHALL with explicit p99 and hard-limit numbers (✅: Req 8).
4. Every circuit breaker from design §10.1 has a SHALL describing its trigger and recovery (✅: Req 9, Req 10).
5. No requirement uses `should`, `would`, `may`, or `could`; every modal verb is `SHALL` (verified by grep on this file).
6. Every requirement traces to at least one design section (✅: every Traceability subsection cites at least one §).
