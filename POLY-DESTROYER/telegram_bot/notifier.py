"""Telegram Bot — live streaming trade notifications + interactive menu.

Push notifications:
  - 🎯 Entry: new trade → then LIVE UPDATES every 1s with prices until resolution
  - 💰 SELL button on live positions — instant market sell with PnL
  - ✅❌ Resolution: win/loss with PnL

Interactive menu (buttons):
  - 📊 Status — SOL, positions, pending, session stats
  - 📌 Positions — open positions detail
  - ⏳ Pending — pending signals detail
  - 📜 History — last 10 trades from JSON log
  - 💰 Capital — capital and PnL breakdown
  - 🔄 Refresh — update status
"""

import os
import re
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from live_trader.ml_shares_trader import MLSharesTrader, MLPosition, TradeResult

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode

from core.utils.logger import log


class TelegramNotifier:
    """Async Telegram bot with live position streaming."""

    def __init__(self, trader: "MLSharesTrader"):
        self.trader = trader
        self._token = os.getenv("TG_BOT_TOKEN", "")
        self._chat_id = os.getenv("TG_CHAT_ID", "")
        self._app: Optional[Application] = None
        self._enabled = bool(self._token and self._chat_id and self._token != "your_bot_token_here")

        # Live streaming: slug → {msg_id, pos, task}
        self._live_msgs: Dict[str, dict] = {}

        if not self._enabled:
            log.warning("  ⚠️ Telegram bot disabled — set TG_BOT_TOKEN & TG_CHAT_ID in .env")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ═══════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════════════

    async def start(self):
        """Start bot polling (non-blocking, runs alongside trader)."""
        if not self._enabled:
            return

        self._app = Application.builder().token(self._token).build()

        # Commands
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("menu", self._cmd_menu))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("pending", self._cmd_pending))
        self._app.add_handler(CommandHandler("history", self._cmd_history))
        self._app.add_handler(CommandHandler("capital", self._cmd_capital))

        # Inline button callbacks
        self._app.add_handler(CallbackQueryHandler(self._on_callback))

        # Suppress noisy network error tracebacks
        import logging as _logging
        _logging.getLogger("httpx").setLevel(_logging.WARNING)
        _logging.getLogger("httpcore").setLevel(_logging.WARNING)
        _logging.getLogger("telegram.ext._updater").setLevel(_logging.WARNING)
        self._app.add_error_handler(self._on_error)

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        log.info("  ✅ Telegram bot started")
        await self._send(self._fmt_startup())

    async def stop(self):
        """Graceful shutdown."""
        # Stop all live streams
        for slug, info in list(self._live_msgs.items()):
            task = info.get("task")
            if task and not task.done():
                task.cancel()
        self._live_msgs.clear()

        if not self._app:
            return
        try:
            await self._send(self._fmt_summary())
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════
    #  PUSH NOTIFICATIONS (called by trader)
    # ═══════════════════════════════════════════════════════════

    async def notify_entry(self, pos: "MLPosition"):
        """Push: new trade → send initial message, start live streaming."""
        if not self._enabled:
            return
        text = self._fmt_entry_live(pos)
        kb = self._sell_keyboard(pos)
        msg = await self._send_and_return(text, markup=kb)
        if msg:
            task = asyncio.create_task(self._stream_position(pos, msg.message_id))
            self._live_msgs[pos.market_slug] = {
                "msg_id": msg.message_id,
                "pos": pos,
                "task": task,
            }

    async def notify_resolution(self, pos: "MLPosition", result: "TradeResult", outcome: str):
        """Push: trade resolved → stop streaming, send final message."""
        if not self._enabled:
            return

        # Stop live stream for this position
        info = self._live_msgs.pop(pos.market_slug, None)
        if info:
            task = info.get("task")
            if task and not task.done():
                task.cancel()
            # Edit the live message to show FINAL result
            try:
                final_text = self._fmt_resolution(pos, result, outcome)
                await self._app.bot.edit_message_text(
                    chat_id=self._chat_id,
                    message_id=info["msg_id"],
                    text=final_text,
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                pass

        # Also send as new message for visibility
        text = self._fmt_resolution(pos, result, outcome)
        await self._send(text)

    # ═══════════════════════════════════════════════════════════
    #  LIVE POSITION STREAMING
    # ═══════════════════════════════════════════════════════════

    async def _stream_position(self, pos: "MLPosition", msg_id: int):
        """Edit entry message every 1s with live prices + SELL button until resolution."""
        _err_count = 0
        try:
            await asyncio.sleep(1.0)
            update_count = 0
            while self.trader._running:
                update_count += 1
                try:
                    text = self._fmt_entry_live(pos, update_count=update_count)
                    kb = self._sell_keyboard(pos)
                    await self._app.bot.edit_message_text(
                        chat_id=self._chat_id,
                        message_id=msg_id,
                        text=text,
                        parse_mode=ParseMode.HTML,
                        reply_markup=kb,
                    )
                    _err_count = 0
                except Exception as e:
                    _err_count += 1
                    err_str = str(e)
                    # "Message is not modified" is normal (data unchanged) — ignore
                    if "not modified" not in err_str.lower():
                        if _err_count <= 3:
                            log.debug(f"TG stream edit #{update_count} error: {err_str[:80]}")
                        if _err_count > 10:
                            log.warning(f"TG stream for {pos.market_slug}: {_err_count} consecutive errors, stopping")
                            break
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    # ═══════════════════════════════════════════════════════════
    #  COMMAND HANDLERS
    # ═══════════════════════════════════════════════════════════

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            self._fmt_welcome(), parse_mode=ParseMode.HTML,
            reply_markup=self._main_keyboard(),
        )

    async def _cmd_menu(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📋 <b>Main Menu</b>", parse_mode=ParseMode.HTML,
            reply_markup=self._main_keyboard(),
        )

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self._fmt_status(), parse_mode=ParseMode.HTML, reply_markup=self._detail_keyboard())

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self._fmt_positions(), parse_mode=ParseMode.HTML, reply_markup=self._detail_keyboard())

    async def _cmd_pending(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self._fmt_pending(), parse_mode=ParseMode.HTML, reply_markup=self._detail_keyboard())

    async def _cmd_history(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self._fmt_history(), parse_mode=ParseMode.HTML, reply_markup=self._detail_keyboard())

    async def _cmd_capital(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self._fmt_capital(), parse_mode=ParseMode.HTML, reply_markup=self._detail_keyboard())

    # ═══════════════════════════════════════════════════════════
    #  INLINE BUTTON CALLBACK
    # ═══════════════════════════════════════════════════════════

    async def _on_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query

        # ── SELL callback: sell:<slug> ──
        if q.data and q.data.startswith("sell:"):
            slug = q.data[5:]
            await q.answer("⏳ Selling...")
            await self._sell_position(slug, q)
            return

        await q.answer()

        formatters = {
            "status": self._fmt_status,
            "positions": self._fmt_positions,
            "pending": self._fmt_pending,
            "history": self._fmt_history,
            "capital": self._fmt_capital,
            "menu": lambda: "📋 <b>Main Menu</b>",
        }

        fmt_fn = formatters.get(q.data)
        if not fmt_fn:
            return

        text = fmt_fn()
        kb = self._main_keyboard() if q.data == "menu" else self._detail_keyboard()

        try:
            await q.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=kb)
        except Exception:
            pass

    async def _sell_position(self, slug: str, q):
        """Execute instant market sell for a position, triggered by SELL button."""
        t = self.trader
        pos = None
        for p in t.positions:
            if p.market_slug == slug:
                pos = p
                break
        if not pos:
            try:
                await q.edit_message_text("❌ Position not found or already closed.", parse_mode=ParseMode.HTML)
            except Exception:
                pass
            return

        # Stop live stream
        info = self._live_msgs.pop(slug, None)
        if info:
            task = info.get("task")
            if task and not task.done():
                task.cancel()

        # Get current bid for PnL calc
        mkt_info = t._active_markets.get(slug, {})
        market = mkt_info.get("market")
        if pos.direction == "UP":
            current_bid = market.best_bid if market and market.best_bid > 0 else pos.entry_price
        else:
            current_bid = market.no_best_bid if market and market.no_best_bid > 0 else pos.entry_price

        # Cancel existing sell limit order first
        if pos.sell_order_id and not t.dry_run and not t._clob.is_read_only:
            try:
                await t._clob.cancel_order(pos.sell_order_id)
            except Exception:
                pass

        # Execute market sell (FAK — immediate fill)
        sell_result = await t._clob.sell_market(
            token_id=pos.token_id,
            shares=pos.shares,
            worst_price=max(0.01, current_bid - 0.03),
            neg_risk=pos.neg_risk,
        )

        exit_price = current_bid if sell_result.success else 0
        pnl_usd = (exit_price - pos.entry_price) * pos.shares
        pnl_pct = (exit_price - pos.entry_price) / max(pos.entry_price, 0.01) * 100
        t._capital += pos.shares * exit_price

        from live_trader.ml_shares_trader import TradeResult
        result = TradeResult(
            slug=slug, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            shares=pos.shares, pnl_usd=pnl_usd, pnl_pct=pnl_pct,
            confidence=pos.confidence, model_prob=pos.model_prob,
            hold_time_s=time.time() - pos.entry_ts,
            reason="manual_sell_win" if pnl_usd >= 0 else "manual_sell_loss",
            sol_at_entry=pos.sol_price_at_entry, sol_at_exit=t._sol_price,
            ptb=pos.price_to_beat, ts=time.time(),
        )
        t.completed.append(result)
        t.positions.remove(pos)

        outcome = "UP" if t._sol_price >= pos.price_to_beat else "DOWN"
        t._log_trade_json(pos, result, outcome)

        emoji = "💰" if pnl_usd >= 0 else "📉"
        mode = "DRY" if t.dry_run else "LIVE"
        hold_s = int(result.hold_time_s)
        hold_m = hold_s // 60
        hold_s = hold_s % 60

        sold_text = (
            f"{emoji} <b>SOLD</b> │ {mode}\n"
            f"<code>{'━' * 28}</code>\n"
            f"<code>{slug}</code>\n\n"
            f"{'📈' if pos.direction == 'UP' else '📉'} {pos.direction} "
            f"${pos.entry_price:.3f} → ${exit_price:.3f}\n"
            f"💵 PnL <b>{pnl_pct:+.0f}%</b> (${pnl_usd:+.2f})\n"
            f"⏱ {hold_m}m {hold_s}s\n\n"
            f"{'✅ Filled' if sell_result.success else '❌ Sell failed'}"
        )

        try:
            await q.edit_message_text(sold_text, parse_mode=ParseMode.HTML)
        except Exception:
            pass

        log.info(
            f"  🔔 TG SELL: {pos.direction} {slug} @ ${exit_price:.3f} "
            f"PnL=${pnl_usd:+.2f} ({pnl_pct:+.0f}%) "
            f"{'OK' if sell_result.success else 'FAILED'}"
        )

    # ═══════════════════════════════════════════════════════════
    #  KEYBOARDS
    # ═══════════════════════════════════════════════════════════

    def _main_keyboard(self):
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("📌 Positions", callback_data="positions"),
            ],
            [
                InlineKeyboardButton("⏳ Pending", callback_data="pending"),
                InlineKeyboardButton("📜 History", callback_data="history"),
            ],
            [
                InlineKeyboardButton("💰 Capital", callback_data="capital"),
            ],
        ])

    def _detail_keyboard(self):
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="status"),
                InlineKeyboardButton("◀️ Menu", callback_data="menu"),
            ],
        ])

    def _sell_keyboard(self, pos: "MLPosition"):
        """Sell button with live PnL shown on the button text."""
        pnl = self._unrealized_pnl(pos)
        pnl_str = f"${pnl:+.2f}" if pnl is not None else "?"
        return InlineKeyboardMarkup([
            [InlineKeyboardButton(f"💰 SELL ({pnl_str})", callback_data=f"sell:{pos.market_slug}")],
        ])

    def _unrealized_pnl(self, pos: "MLPosition") -> float:
        """Calculate unrealized PnL based on current bid price."""
        t = self.trader
        mkt_info = t._active_markets.get(pos.market_slug, {})
        market = mkt_info.get("market")
        if not market:
            return 0.0
        if pos.direction == "UP":
            bid = market.best_bid if market.best_bid > 0 else pos.entry_price
        else:
            bid = market.no_best_bid if market.no_best_bid > 0 else pos.entry_price
        return (bid - pos.entry_price) * pos.shares

    # ═══════════════════════════════════════════════════════════
    #  MESSAGE FORMATTERS
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _bar(pct: float, w: int = 10) -> str:
        pct = max(0.0, min(1.0, pct))
        f = int(pct * w)
        return '█' * f + '░' * (w - f)

    @staticmethod
    def _thin_bar(pct: float, w: int = 12) -> str:
        pct = max(0.0, min(1.0, pct))
        f = int(pct * w)
        return '▓' * f + '░' * (w - f)

    def _model_sets_info(self) -> str:
        sets = getattr(self.trader, '_model_sets', {})
        if not sets:
            return "1 source"
        return ' + '.join(f"{s}({len(m.get('models', {}))})" for s, m in sets.items())

    def _sol_line(self) -> str:
        t = self.trader
        bn = getattr(t, '_binance_sol', 0)
        s = f"${t._sol_price:.2f}"
        if bn > 0:
            s += f" (Bin ${bn:.2f} {t._sol_price - bn:+.3f})"
        return s

    def _liq_line(self) -> str:
        liq = getattr(self.trader, '_liq_recorder', None)
        if not liq or liq._total_events == 0:
            return ""
        p = liq.get_pressure(self.trader._sol_price, radius=2.0)
        imb = p["imbalance"]
        icon = "🔻" if imb > 0 else "🔺" if imb < 0 else "➖"
        return f"💀 {liq._total_events} liqs {icon} ${imb:+,.0f} │ L${p['long_usd']:,.0f} S${p['short_usd']:,.0f}"

    def _models_block(self, probs: dict) -> str:
        if not probs:
            return ""
        sources = {}
        for mn, p in sorted(probs.items()):
            parts = mn.split('_', 1)
            src = parts[0] if len(parts) > 1 else "?"
            name = parts[1] if len(parts) > 1 else mn
            sources.setdefault(src, []).append((name, p))
        lines = []
        for src, models in sources.items():
            icon = "🅱️" if src == "binance" else "🅿️" if src in ("hermes", "pyth") else "🔹"
            lines.append(f"{icon} <b>{src.upper()}</b>")
            for name, p in models:
                dp = max(p, 1 - p)
                d = "▲" if p > 0.5 else "▼"
                lines.append(f"<code>  {name:<9} {d} {dp*100:>3.0f}% {self._bar(dp, 8)}</code>")
        return "\n".join(lines)

    def _fmt_startup(self) -> str:
        t = self.trader
        mode = "🔵 DRY RUN" if t.dry_run else "🔴 LIVE"
        n = sum(len(s.get('models', {})) for s in getattr(t, '_model_sets', {}).values())
        src = list(getattr(t, '_model_sets', {}).keys())
        streak = getattr(t, 'streak_required', 1)
        streak_str = f"{streak}x" if streak > 1 else "off"
        return (
            f"<code>╔══════════════════════════════╗\n"
            f"║   🤖 POLY-DESTROYER          ║\n"
            f"╚══════════════════════════════╝</code>\n"
            f"{mode}\n\n"
            f"⚙️ <b>Config</b>\n"
            f"<code>  Model     {t.primary_model_name}\n"
            f"  Conf      ≥ {t.min_confidence:.0%}\n"
            f"  Streak    {streak_str} ≥ {t.min_confidence:.0%}\n"
            f"  Shares    ${t.min_share_price:.2f}–${t.max_share_price:.2f}\n"
            f"  Order     ${t.order_size:.2f} × {t.max_positions}\n"
            f"  Models    {n} ({', '.join(src)})</code>\n"
            f"\n⚡ ~8ms │ 📡 Pyth + Binance │ 🔄 1s TG"
        )

    def _fmt_welcome(self) -> str:
        return (
            "<code>╔══════════════════════════════╗\n"
            "║   🤖 POLY-DESTROYER          ║\n"
            "╚══════════════════════════════╝</code>\n\n"
            "ML trading on Polymarket\n"
            "Pyth Hermes + Binance dual oracle\n\n"
            "<b>Commands</b>\n"
            "<code>/status     📊 dashboard\n"
            "/positions  📌 open trades\n"
            "/pending    ⏳ signals\n"
            "/history    📜 trades\n"
            "/capital    💰 PnL</code>"
        )

    def _fmt_entry_live(self, pos: "MLPosition", update_count: int = 0) -> str:
        """Live-updating entry message. Called every 1s by _stream_position."""
        t = self.trader
        mode = "DRY" if t.dry_run else "LIVE"
        arrow = "📈" if pos.direction == "UP" else "📉"

        sol_now = t._sol_price
        ptb = pos.price_to_beat
        delta_now = sol_now - ptb
        winning = (pos.direction == "UP" and delta_now > 0) or (pos.direction == "DOWN" and delta_now < 0)
        status = "🟢" if winning else "🔴"

        secs_left = self._secs_left(pos.end_date, slug=pos.market_slug, duration_minutes=pos.duration_minutes)
        elapsed_pct = 1.0 - (secs_left / (pos.duration_minutes * 60)) if pos.duration_minutes else 0
        tbar = self._thin_bar(elapsed_pct, 12)
        tl = f"{int(secs_left // 60)}:{int(secs_left % 60):02d}" if secs_left >= 60 else f"{int(secs_left)}s"

        info = t._active_markets.get(pos.market_slug, {})
        market = info.get("market")
        if pos.direction == "UP":
            cur_bid = market.best_bid if market and market.best_bid > 0 else pos.entry_price
            cur_ask = market.best_ask if market and market.best_ask > 0 else 0
        else:
            cur_bid = market.no_best_bid if market and market.no_best_bid > 0 else pos.entry_price
            cur_ask = market.no_best_ask if market and market.no_best_ask > 0 else 0
        spread = (cur_ask - cur_bid) if cur_ask > 0 and cur_bid > 0 else 0

        # Live PnL
        pnl_usd = (cur_bid - pos.entry_price) * pos.shares
        pnl_pct = (cur_bid - pos.entry_price) / max(pos.entry_price, 0.01) * 100
        pnl_icon = "🟢" if pnl_usd >= 0 else "🔴"
        # If resolved at $1 (win scenario)
        max_pnl = (1.0 - pos.entry_price) * pos.shares

        bn = getattr(t, '_binance_sol', 0)

        probs = getattr(pos, 'all_model_probs', {})
        inf_ms = getattr(pos, 'inference_ms', 0)

        dot = "●" if update_count % 2 == 0 else "○"

        lines = [
            f"{arrow} <b>{pos.direction}</b> │ {mode} │ {status} {'WIN' if winning else 'LOSE'}",
            f"<code>{'━' * 28}</code>",
            f"<code>{pos.market_slug}</code>",
            f"<code>{tbar} {tl}</code>",
            f"",
            f"💰 ${pos.entry_price:.3f} × {pos.shares:.1f} = ${pos.size_usd:.2f}",
            f"🧠 {pos.confidence:.0%} │ ⚡ {inf_ms:.0f}ms",
            f"",
            f"{pnl_icon} PnL <b>${pnl_usd:+.2f}</b> ({pnl_pct:+.0f}%) │ max ${max_pnl:.2f}",
            f"📊 bid ${cur_bid:.3f} ask ${cur_ask:.3f} sp ${spread:.3f}",
            f"",
            f"<code>┌─ SOL ─────────────┐</code>",
            f"<code>│ ${sol_now:.4f}{'  Bin $'+f'{bn:.2f}' if bn > 0 else '':<14}│</code>",
            f"<code>│ PTB ${ptb:.4f}        │</code>",
            f"<code>│ {status} Δ {delta_now:+.4f}{'':>8}│</code>",
            f"<code>└───────────────────┘</code>",
        ]

        if probs:
            lines.append("")
            lines.append(self._models_block(probs))

        liq = self._liq_line()
        if liq:
            lines.append(f"\n{liq}")

        if update_count > 0:
            lines.append(f"\n{dot} <i>#{update_count}</i>")

        return "\n".join(lines)

    def _fmt_resolution(self, pos: "MLPosition", result: "TradeResult", outcome: str) -> str:
        won = "win" in result.reason
        emoji = "✅" if won else "❌"
        label = "WIN" if won else "LOSS"
        gap_entry = pos.sol_price_at_entry - pos.price_to_beat
        gap_exit = result.sol_at_exit - result.ptb

        t = self.trader
        wins = sum(1 for tr in t.completed if "win" in tr.reason)
        total = len(t.completed)
        losses = total - wins
        wr = (wins / total * 100) if total else 0
        pnl = sum(tr.pnl_usd for tr in t.completed)
        hold_m = int(result.hold_time_s // 60)
        hold_s = int(result.hold_time_s % 60)

        streak = 0
        for tr in reversed(t.completed):
            if ("win" in tr.reason) == won:
                streak += 1
            else:
                break
        streak_line = f"{'🔥' if won else '💀'} <b>{streak}x</b>\n" if streak >= 2 else ""

        probs = getattr(pos, 'all_model_probs', {})
        n_agree = sum(1 for p in probs.values() if (p > 0.5) == (pos.direction == "UP")) if probs else 0

        return (
            f"{emoji} <b>{label}</b> │ {pos.direction} → {outcome}\n"
            f"<code>{'━' * 28}</code>\n"
            f"<code>{pos.market_slug}</code>\n\n"
            f"💰 PnL <b>{result.pnl_pct:+.0f}%</b> (${result.pnl_usd:+.2f}) {self._bar(min(abs(result.pnl_pct)/100, 1), 10)}\n"
            f"🧠 {pos.confidence:.0%} │ {n_agree}/{len(probs)} agreed\n"
            f"📊 SOL ${pos.sol_price_at_entry:.2f} → ${result.sol_at_exit:.2f}\n"
            f"📐 Gap {gap_entry:+.3f} → {gap_exit:+.3f}\n"
            f"⏱ {hold_m}m {hold_s}s\n\n"
            f"<code>{'━' * 28}</code>\n"
            f"🏆 {wins}W {losses}L │ {wr:.0f}% {self._bar(wr/100, 10) if total else ''}\n"
            f"💵 <b>${t._capital:.2f}</b> │ ${pnl:+.2f}\n"
            f"{streak_line}"
        )

    def _fmt_status(self) -> str:
        t = self.trader
        elapsed = time.time() - t._start_ts
        wins = sum(1 for tr in t.completed if "win" in tr.reason)
        total = len(t.completed)
        losses = total - wins
        wr = (wins / total * 100) if total else 0
        pnl = sum(tr.pnl_usd for tr in t.completed)
        cap_chg = (t._capital / 100 - 1) * 100
        mode = "DRY" if t.dry_run else "LIVE"

        lines = [
            f"<code>╔═ 📊 {mode} ═══ {int(elapsed // 60)}m ══════════╗</code>",
            f"",
            f"📈 SOL {self._sol_line()}",
            f"💵 <b>${t._capital:.2f}</b> ({cap_chg:+.1f}%) │ ${pnl:+.2f}",
            f"🏆 {wins}W {losses}L {wr:.0f}% {self._bar(wr/100, 8) if total else ''}",
            f"📡 {len(t._active_markets)} mkts │ {t._clob_updates:,} WS │ {self._model_sets_info()}",
        ]
        liq = self._liq_line()
        if liq:
            lines.append(liq)

        lines.append(f"\n<b>📌 Positions ({len(t.positions)})</b>")
        if t.positions:
            for pos in t.positions:
                delta = t._sol_price - pos.price_to_beat
                w = (pos.direction == "UP" and delta > 0) or (pos.direction == "DOWN" and delta < 0)
                secs = self._secs_left(pos.end_date, slug=pos.market_slug, duration_minutes=pos.duration_minutes)
                tl = f"{secs // 60:.0f}m" if secs else "?"
                lines.append(
                    f"<code>{'🟢' if w else '🔴'} {pos.direction:<4} {self._short_slug(pos.market_slug):<8} "
                    f"${pos.entry_price:.3f} {pos.confidence:.0%} Δ{delta:+.3f} {tl}</code>"
                )
        else:
            lines.append("  <i>none</i>")

        lines.append(f"\n<b>⏳ Pending ({len(t._pending_signals)})</b>")
        if t._pending_signals:
            for slug in t._pending_signals:
                info = t._active_markets.get(slug, {})
                market = info.get("market")
                if not market:
                    continue
                cur = market.best_ask if market.best_ask > 0 else market.yes_price
                lines.append(f"<code>⏳ {self._short_slug(slug):<8} ${cur:.3f} → ≤${t.max_share_price:.2f}</code>")
        else:
            lines.append("  <i>none</i>")
        return "\n".join(lines)

    def _fmt_positions(self) -> str:
        t = self.trader
        if not t.positions:
            return "📌 <b>Positions</b>\n\n<i>No open positions</i>"
        lines = [f"📌 <b>Positions ({len(t.positions)})</b>", ""]
        for pos in t.positions:
            delta = t._sol_price - pos.price_to_beat
            w = (pos.direction == "UP" and delta > 0) or (pos.direction == "DOWN" and delta < 0)
            secs = self._secs_left(pos.end_date, slug=pos.market_slug, duration_minutes=pos.duration_minutes)
            tl = f"{secs // 60:.0f}m {secs % 60:.0f}s" if secs else "?"
            ep = 1.0 - (secs / (pos.duration_minutes * 60)) if pos.duration_minutes and secs else 1.0
            info = t._active_markets.get(pos.market_slug, {})
            market = info.get("market")
            up = market.best_ask if market and market.best_ask > 0 else 0
            dn = market.no_best_ask if market and market.no_best_ask > 0 else 0
            lines.extend([
                f"{'🟢' if w else '🔴'} <b>{pos.direction}</b> <code>{pos.market_slug}</code>",
                f"<code>  ${pos.entry_price:.3f} × {pos.shares:.1f} = ${pos.size_usd:.2f}</code>",
                f"<code>  SOL ${pos.sol_price_at_entry:.2f} → ${t._sol_price:.2f} Δ{delta:+.4f}</code>",
                f"<code>  UP ${up:.3f} DN ${dn:.3f}</code>" if up > 0 else "",
                f"<code>  {self._thin_bar(ep, 10)} {tl}</code>",
                "",
            ])
        return "\n".join(l for l in lines if l is not None)

    def _fmt_pending(self) -> str:
        t = self.trader
        if not t._pending_signals:
            return "⏳ <b>Pending</b>\n\n<i>No pending</i>"
        lines = [f"⏳ <b>Pending ({len(t._pending_signals)})</b>", ""]
        for slug, sig in t._pending_signals.items():
            info = t._active_markets.get(slug, {})
            market = info.get("market")
            if not market:
                continue
            d = sig.get("direction", "?")
            cur = (market.best_ask if d == "UP" else market.no_best_ask) or market.yes_price
            secs = self._secs_left("", slug=slug, duration_minutes=market.duration_minutes)
            tl = f"{secs // 60:.0f}m" if secs else "?"
            lines.append(f"<code>{'📈' if d == 'UP' else '�'} {d:<4} {self._short_slug(slug):<8} ${cur:.3f} → ≤${t.max_share_price:.2f} {tl}</code>")
        return "\n".join(lines)

    def _fmt_history(self) -> str:
        json_path = Path("results/ml_live_trades.json")
        if not json_path.exists():
            return "📜 <b>History</b>\n\n<i>No trades yet</i>"
        try:
            trades = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return "📜 <b>History</b>\n\n<i>Error</i>"
        if not trades:
            return "📜 <b>History</b>\n\n<i>No trades</i>"
        recent = list(reversed(trades[-10:]))
        lines = [f"📜 <b>History</b> ({len(recent)}/{len(trades)})", ""]
        for tr in recent:
            won = tr.get("won", 0)
            e = "✅" if won else "❌"
            d = tr.get("direction", "?")
            pnl_pct = tr.get("pnl_pct", 0)
            pnl_usd = tr.get("pnl_usd", 0)
            conf = tr.get("confidence", 0) * 100
            t_str = tr.get("exit_time", "?")[-8:]
            lines.append(f"<code>{e} {d:<4} {self._short_slug(tr.get('slug', '')):<8} {conf:>3.0f}% {pnl_pct:>+4.0f}% ${pnl_usd:>+5.2f} {t_str}</code>")
        aw = sum(1 for x in trades if x.get("won"))
        an = len(trades)
        ap = sum(x.get("pnl_usd", 0) for x in trades)
        lines.extend(["", f"<b>{aw}W/{an-aw}L ({aw/an*100 if an else 0:.0f}%) │ ${ap:+.2f}</b>"])
        return "\n".join(lines)

    def _fmt_capital(self) -> str:
        t = self.trader
        elapsed = time.time() - t._start_ts
        pnl = sum(tr.pnl_usd for tr in t.completed)
        wins = sum(1 for tr in t.completed if "win" in tr.reason)
        total = len(t.completed)
        losses = total - wins
        wr = (wins / total * 100) if total else 0
        cap_chg = (t._capital / 100 - 1) * 100
        lines = [
            f"<code>╔═ 💰 CAPITAL ══════════════╗</code>",
            f"",
            f"🏦 $100 → <b>${t._capital:.2f}</b> ({cap_chg:+.1f}%)",
            f"<code>   {self._bar(min(t._capital/100, 2) / 2, 14)}</code>",
            f"📈 PnL: <b>${pnl:+.2f}</b>",
            f"🏆 {total} trades │ {wins}W {losses}L │ {wr:.0f}%",
        ]
        if wins:
            lines.append(f"✅ Avg win: ${sum(tr.pnl_usd for tr in t.completed if 'win' in tr.reason)/wins:+.2f}")
        if losses:
            lines.append(f"❌ Avg loss: ${sum(tr.pnl_usd for tr in t.completed if 'loss' in tr.reason)/losses:+.2f}")
        lines.extend([
            f"",
            f"⏱ {int(elapsed // 3600)}h {int(elapsed % 3600 // 60)}m │ 📡 {t._clob_updates:,} WS",
        ])
        return "\n".join(lines)

    def _fmt_summary(self) -> str:
        t = self.trader
        if not t.completed:
            return "🛑 <b>Session ended</b> — no trades."
        elapsed = time.time() - t._start_ts
        wins = sum(1 for tr in t.completed if "win" in tr.reason)
        total = len(t.completed)
        losses = total - wins
        wr = (wins / total * 100) if total else 0
        pnl = sum(tr.pnl_usd for tr in t.completed)
        cap_chg = (t._capital / 100 - 1) * 100
        return (
            f"<code>╔══════════════════════════════╗\n"
            f"║   🛑 SESSION ENDED            ║\n"
            f"╚══════════════════════════════╝</code>\n\n"
            f"⏱ {int(elapsed // 3600)}h {int(elapsed % 3600 // 60)}m\n"
            f"🏆 {wins}W {losses}L │ {wr:.0f}% {self._bar(wr/100, 10)}\n"
            f"💰 <b>${pnl:+.2f}</b>\n"
            f"💵 $100 → <b>${t._capital:.2f}</b> ({cap_chg:+.1f}%)"
        )

    # ═══════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════

    def _short_slug(self, slug: str) -> str:
        """sol-updown-5m-1777821000 → 5m-...1000"""
        parts = slug.split("-")
        if len(parts) >= 4:
            dur = parts[2]  # 5m or 15m
            ts = parts[3][-4:]  # last 4 digits
            return f"{dur}-{ts}"
        return slug[-12:]

    def _secs_left(self, end_date: str, slug: str = "", duration_minutes: int = 0) -> float:
        # Priority 1: compute from slug epoch (most reliable)
        if slug and duration_minutes:
            import re
            ts_m = re.search(r'-(\d{10})$', slug)
            if ts_m:
                end_epoch = int(ts_m.group(1)) + duration_minutes * 60
                return max(0, end_epoch - time.time())
        # Priority 2: parse end_date ISO string
        if not end_date:
            return 0
        try:
            end_ts = datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp()
            return max(0, end_ts - time.time())
        except (ValueError, TypeError):
            return 0

    @staticmethod
    async def _on_error(update, context):
        """Silently handle network errors (DNS failures, timeouts, etc.)."""
        import telegram.error
        if isinstance(context.error, telegram.error.NetworkError):
            log.debug(f"TG network error (will retry): {context.error}")
        else:
            log.warning(f"TG error: {context.error}")

    async def _send_and_return(self, text: str, markup=None):
        """Send message and return the Message object (for live streaming edits)."""
        if not self._enabled or not self._app:
            return None
        for attempt in range(3):
            try:
                return await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
            except Exception as e:
                log.debug(f"TG send_and_return error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
        return None

    async def _send(self, text: str, markup=None):
        """Send message to configured chat."""
        if not self._enabled or not self._app:
            return
        for attempt in range(3):
            try:
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
                return
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    log.warning(f"TG send failed (loop closed): {e}")
                    return
                log.warning(f"TG send error (attempt {attempt+1}): {e}")
            except Exception as e:
                log.warning(f"TG send error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1)

    async def _send_photo(self, photo_bytes: bytes, caption: str = ""):
        """Send a photo (PNG bytes) to configured chat."""
        if not self._enabled or not self._app:
            return
        for attempt in range(3):
            try:
                import io
                await self._app.bot.send_photo(
                    chat_id=self._chat_id,
                    photo=io.BytesIO(photo_bytes),
                    caption=caption,
                    parse_mode=ParseMode.HTML,
                )
                return
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    log.warning(f"TG photo send failed (loop closed): {e}")
                    return
                log.warning(f"TG photo send error (attempt {attempt+1}): {e}")
            except Exception as e:
                log.warning(f"TG photo send error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1)

    async def send_trade_card(self, pos: "MLPosition", result: "TradeResult", outcome: str):
        """Render and send a PnL card image after trade resolution."""
        if not self._enabled:
            return
        try:
            from cards.renderer import render_trade_card
            trade_data = {
                "slug": pos.market_slug,
                "direction": pos.direction,
                "won": "win" in result.reason,
                "pnl_usd": result.pnl_usd,
                "pnl_pct": result.pnl_pct,
                "confidence": pos.confidence,
                "entry_price": pos.entry_price,
                "exit_price": result.exit_price,
                "shares": pos.shares,
                "size_usd": pos.size_usd,
                "hold_time_s": result.hold_time_s,
                "sol_at_entry": pos.sol_price_at_entry,
                "sol_at_exit": result.sol_at_exit,
                "ptb": pos.price_to_beat,
                "primary_model": getattr(pos, 'primary_model', 'lgbm'),
                "all_model_probs": getattr(pos, 'all_model_probs', {}),
                "dry_run": self.trader.dry_run,
                "exit_time": datetime.now().strftime("%H:%M:%S"),
            }
            card_bytes = render_trade_card(trade_data)
            won = "win" in result.reason
            caption = f"{'✅' if won else '❌'} {pos.direction} | {result.pnl_pct:+.0f}% (${result.pnl_usd:+.2f})"
            await self._send_photo(card_bytes, caption)
        except Exception as e:
            log.warning(f"Trade card render/send error: {e}")
