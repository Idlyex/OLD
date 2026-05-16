"""Telegram Bot — beautiful live trade notifications + interactive menu.

Push notifications:
  - 🎯 Entry: every new trade
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
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, TYPE_CHECKING

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
    """Async Telegram bot for ML Shares Trader."""

    def __init__(self, trader: "MLSharesTrader"):
        self.trader = trader
        self._token = os.getenv("TG_BOT_TOKEN", "")
        self._chat_id = os.getenv("TG_CHAT_ID", "")
        self._app: Optional[Application] = None
        self._enabled = bool(self._token and self._chat_id and self._token != "your_bot_token_here")

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

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        log.info("  ✅ Telegram bot started")
        await self._send(self._fmt_startup())

    async def stop(self):
        """Graceful shutdown."""
        if not self._app:
            return
        try:
            # Send final summary
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
        """Push: new trade opened."""
        if not self._enabled:
            return
        text = self._fmt_entry(pos)
        await self._send(text)

    async def notify_resolution(self, pos: "MLPosition", result: "TradeResult", outcome: str):
        """Push: trade resolved."""
        if not self._enabled:
            return
        text = self._fmt_resolution(pos, result, outcome)
        await self._send(text)

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
        await update.message.reply_text(self._fmt_status(), parse_mode=ParseMode.HTML)

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self._fmt_positions(), parse_mode=ParseMode.HTML)

    async def _cmd_pending(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self._fmt_pending(), parse_mode=ParseMode.HTML)

    async def _cmd_history(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self._fmt_history(), parse_mode=ParseMode.HTML)

    async def _cmd_capital(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(self._fmt_capital(), parse_mode=ParseMode.HTML)

    # ═══════════════════════════════════════════════════════════
    #  INLINE BUTTON CALLBACK
    # ═══════════════════════════════════════════════════════════

    async def _on_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
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
            pass  # message unchanged

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

    # ═══════════════════════════════════════════════════════════
    #  MESSAGE FORMATTERS
    # ═══════════════════════════════════════════════════════════

    def _fmt_startup(self) -> str:
        t = self.trader
        mode = "🔵 DRY RUN" if t.dry_run else "🔴 LIVE"
        return (
            "🤖 <b>ML Shares Trader Online</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  {mode}\n"
            f"  🎯 Confidence: ≥{t.min_confidence:.0%}\n"
            f"  💲 Max share: ${t.max_share_price:.2f}\n"
            f"  💰 Order size: ${t.order_size:.2f}\n"
            f"  📊 Max positions: {t.max_positions}\n"
            f"  🔍 Slugs: {', '.join(t.slugs)}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    def _fmt_welcome(self) -> str:
        return (
            "👋 <b>ML Shares Trader Bot</b>\n\n"
            "Real-time notifications for:\n"
            "  🎯 New trades\n"
            "  ✅ Wins / ❌ Losses\n\n"
            "Use the buttons below or commands:\n"
            "  /status — live dashboard\n"
            "  /positions — open trades\n"
            "  /pending — waiting for price\n"
            "  /history — last 10 trades\n"
            "  /capital — PnL breakdown\n"
        )

    def _fmt_entry(self, pos: "MLPosition") -> str:
        mode = "🔵 DRY" if self.trader.dry_run else "🔴 LIVE"
        arrow = "📈" if pos.direction == "UP" else "📉"
        gap = pos.sol_price_at_entry - pos.price_to_beat
        gap_pct = gap / pos.price_to_beat * 100 if pos.price_to_beat else 0

        secs_left = self._secs_left(pos.end_date, slug=pos.market_slug, duration_minutes=pos.duration_minutes)
        time_str = f"{secs_left // 60:.0f}m {secs_left % 60:.0f}s" if secs_left else "?"

        # Per-model probabilities
        model_lines = ""
        probs = getattr(pos, 'all_model_probs', {})
        if probs:
            for mn, p in sorted(probs.items()):
                dp = max(p, 1 - p) * 100
                d = "UP" if p > 0.5 else "DN"
                model_lines += f"    {mn:<10} {d} {dp:.0f}%\n"

        primary = getattr(pos, 'primary_model', 'lgbm')
        inf_ms = getattr(pos, 'inference_ms', 0)

        return (
            f"🎯 <b>NEW TRADE</b> — {arrow} {pos.direction}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  📋 <code>{pos.market_slug}</code>\n"
            f"  ⏱ {pos.duration_minutes}m market  |  ~{time_str} left\n"
            "\n"
            f"  💰 ${pos.entry_price:.3f} × {pos.shares:.1f} = <b>${pos.size_usd:.2f}</b>\n"
            f"  🧠 Confidence: <b>{pos.confidence:.0%}</b>  (primary: {primary})\n"
            f"  ⚡ ML inference: {inf_ms:.0f}ms\n"
            "\n"
            f"  📊 SOL ${pos.sol_price_at_entry:.2f}  |  PTB ${pos.price_to_beat:.2f}\n"
            f"  📐 Gap: {gap:+.4f} ({gap_pct:+.2f}%)\n"
            "\n"
            f"  🤖 <b>Models:</b>\n"
            f"<pre>{model_lines}</pre>"
            f"  {mode}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    def _fmt_resolution(self, pos: "MLPosition", result: "TradeResult", outcome: str) -> str:
        won = "win" in result.reason
        emoji = "✅" if won else "❌"
        label = "WIN" if won else "LOSS"
        arrow = "📈" if pos.direction == "UP" else "📉"
        gap_entry = pos.sol_price_at_entry - pos.price_to_beat
        gap_exit = result.sol_at_exit - result.ptb
        gap_pct = gap_entry / pos.price_to_beat * 100 if pos.price_to_beat else 0

        # Session stats
        t = self.trader
        wins = sum(1 for tr in t.completed if "win" in tr.reason)
        total = len(t.completed)
        losses = total - wins
        wr = (wins / total * 100) if total else 0
        pnl = sum(tr.pnl_usd for tr in t.completed)

        hold_m = int(result.hold_time_s // 60)
        hold_s = int(result.hold_time_s % 60)

        # Streak
        streak = 0
        for tr in reversed(t.completed):
            if ("win" in tr.reason) == won:
                streak += 1
            else:
                break
        streak_str = f"{'🔥' if won else '💀'} {streak}x streak" if streak >= 2 else ""

        # Per-model agreement
        probs = getattr(pos, 'all_model_probs', {})
        n_agree = sum(1 for p in probs.values() if (p > 0.5) == (pos.direction == "UP")) if probs else 0
        n_total = len(probs)
        consensus = f"{n_agree}/{n_total} models agreed" if n_total else ""

        primary = getattr(pos, 'primary_model', 'lgbm')

        return (
            f"{emoji} <b>{label}</b> — {arrow} {pos.direction} → {outcome}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  📋 <code>{pos.market_slug}</code>\n"
            "\n"
            f"  💰 <b>{result.pnl_pct:+.0f}%</b>  (${result.pnl_usd:+.2f})\n"
            f"  🧠 {pos.confidence:.0%} confident  ({primary})\n"
            f"  🤖 {consensus}\n"
            f"  📊 SOL ${pos.sol_price_at_entry:.2f} → ${result.sol_at_exit:.2f}\n"
            f"  📐 Gap: {gap_entry:+.4f} ({gap_pct:+.2f}%) → {gap_exit:+.4f}\n"
            f"  ⏱ Held {hold_m}m {hold_s}s\n"
            f"  💵 Capital: <b>${t._capital:.2f}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  🏆 {wins}W / {losses}L ({wr:.0f}%)  |  ${pnl:+.2f}\n"
            f"  {streak_str}\n"
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

        lines = [
            f"📊 <b>LIVE STATUS</b>  ·  {int(elapsed // 60)}m uptime",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  📈 SOL: <b>${t._sol_price:.2f}</b>",
            f"  💵 Capital: <b>${t._capital:.2f}</b> ({cap_chg:+.1f}%)",
            f"  🏆 {wins}W / {losses}L ({wr:.0f}%)  |  PnL ${pnl:+.2f}",
            f"  📡 Markets: {len(t._active_markets)}  |  CLOB: {t._clob_updates}",
            "",
        ]

        # Open positions
        lines.append(f"  📌 <b>Open: {len(t.positions)}</b>")
        if t.positions:
            for pos in t.positions:
                delta = t._sol_price - pos.price_to_beat
                ok = "✓" if (pos.direction == "UP" and delta > 0) or (pos.direction == "DOWN" and delta < 0) else "✗"
                secs = self._secs_left(pos.end_date, slug=pos.market_slug, duration_minutes=pos.duration_minutes)
                tl = f"{secs // 60:.0f}m" if secs else "?"
                lines.append(
                    f"    {'📈' if pos.direction == 'UP' else '📉'} {pos.direction} "
                    f"<code>{self._short_slug(pos.market_slug)}</code> "
                    f"@ ${pos.entry_price:.3f} | {pos.confidence:.0%} | "
                    f"Δ{delta:+.2f} {ok} | {tl}"
                )
        else:
            lines.append("    — none —")

        # Pending
        lines.append(f"\n  ⏳ <b>Pending: {len(t._pending_signals)}</b>")
        if t._pending_signals:
            for slug, sig in t._pending_signals.items():
                info = t._active_markets.get(slug, {})
                market = info.get("market")
                if market:
                    cur = market.yes_price if sig["direction"] == "UP" else market.no_price
                    lines.append(
                        f"    {'📈' if sig['direction'] == 'UP' else '📉'} {sig['direction']} "
                        f"<code>{self._short_slug(slug)}</code> "
                        f"| ${cur:.3f} (need ≤${t.max_share_price:.2f}) | {sig.get('dir_prob', 0):.0%}"
                    )
        else:
            lines.append("    — none —")

        lines.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    def _fmt_positions(self) -> str:
        t = self.trader
        if not t.positions:
            return "📌 <b>Open Positions</b>\n\n— No open positions —"

        lines = [f"📌 <b>Open Positions ({len(t.positions)})</b>", ""]
        for pos in t.positions:
            delta = t._sol_price - pos.price_to_beat
            ok = "✓" if (pos.direction == "UP" and delta > 0) or (pos.direction == "DOWN" and delta < 0) else "✗"
            secs = self._secs_left(pos.end_date, slug=pos.market_slug, duration_minutes=pos.duration_minutes)
            tl = f"{secs // 60:.0f}m {secs % 60:.0f}s" if secs else "?"
            arrow = "📈" if pos.direction == "UP" else "📉"

            lines.extend([
                f"{arrow} <b>{pos.direction}</b>  —  <code>{pos.market_slug}</code>",
                f"  💰 ${pos.entry_price:.3f} × {pos.shares:.1f} = ${pos.size_usd:.2f}",
                f"  🧠 Confidence: {pos.confidence:.0%}",
                f"  📊 SOL ${pos.sol_price_at_entry:.2f} → ${t._sol_price:.2f}",
                f"  📐 Δ = {delta:+.4f} {ok}",
                f"  ⏱ {tl} left",
                "",
            ])
        return "\n".join(lines)

    def _fmt_pending(self) -> str:
        t = self.trader
        if not t._pending_signals:
            return "⏳ <b>Pending Signals</b>\n\n— No pending signals —"

        lines = [f"⏳ <b>Pending Signals ({len(t._pending_signals)})</b>", ""]
        for slug, sig in t._pending_signals.items():
            info = t._active_markets.get(slug, {})
            market = info.get("market")
            if not market:
                continue
            cur = market.yes_price if sig["direction"] == "UP" else market.no_price
            arrow = "📈" if sig["direction"] == "UP" else "📉"
            total_ms = market.time_remaining_ms + market.time_elapsed_ms
            pct = market.time_elapsed_ms / max(total_ms, 1) * 100

            lines.extend([
                f"{arrow} <b>{sig['direction']}</b>  —  <code>{slug}</code>",
                f"  💲 Current: ${cur:.3f}  (need ≤${t.max_share_price:.2f})",
                f"  🧠 Confidence: {sig.get('dir_prob', 0):.0%}",
                f"  📐 PTB gap: {sig.get('sol_at_eval', 0) - sig.get('ptb', 0):+.4f}",
                f"  ⏱ {pct:.0f}% elapsed",
                "",
            ])
        return "\n".join(lines)

    def _fmt_history(self) -> str:
        """Last 10 trades from JSON log."""
        json_path = Path("results/ml_live_trades.json")
        if not json_path.exists():
            return "📜 <b>Trade History</b>\n\n— No trades yet —"

        try:
            trades = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return "📜 <b>Trade History</b>\n\n— Error reading trades —"

        if not trades:
            return "📜 <b>Trade History</b>\n\n— No trades yet —"

        # Last 10
        recent = trades[-10:]
        recent.reverse()  # newest first

        lines = [f"📜 <b>Trade History</b> (last {len(recent)} of {len(trades)})", ""]

        for tr in recent:
            won = tr.get("won", 0)
            emoji = "✅" if won else "❌"
            arrow = "📈" if tr["direction"] == "UP" else "📉"
            conf = tr.get("confidence", 0) * 100
            pnl = tr.get("pnl_usd", 0)
            pnl_pct = tr.get("pnl_pct", 0)
            t_str = tr.get("exit_time", "?")[-8:]  # HH:MM:SS
            gap = tr.get("gap_at_entry", 0)
            slug_short = self._short_slug(tr.get("slug", ""))

            lines.append(
                f"{emoji} {arrow} {tr['direction']} <code>{slug_short}</code> "
                f"| {conf:.0f}% | {pnl_pct:+.0f}% (${pnl:+.2f}) "
                f"| gap {gap:+.3f} | {t_str}"
            )

        # Summary
        all_wins = sum(1 for t in trades if t.get("won"))
        all_total = len(trades)
        all_pnl = sum(t.get("pnl_usd", 0) for t in trades)
        wr = all_wins / all_total * 100 if all_total else 0
        lines.extend([
            "",
            f"📊 <b>All time:</b> {all_wins}W/{all_total - all_wins}L ({wr:.0f}%) | ${all_pnl:+.2f}",
        ])

        return "\n".join(lines)

    def _fmt_capital(self) -> str:
        t = self.trader
        elapsed = time.time() - t._start_ts
        cap_chg = (t._capital / 100 - 1) * 100
        pnl = sum(tr.pnl_usd for tr in t.completed)
        wins = sum(1 for tr in t.completed if "win" in tr.reason)
        total = len(t.completed)
        losses = total - wins

        lines = [
            "💰 <b>Capital Report</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  💵 Start:   $100.00",
            f"  💵 Current: <b>${t._capital:.2f}</b> ({cap_chg:+.1f}%)",
            f"  📈 PnL:     <b>${pnl:+.2f}</b>",
            "",
            f"  🏆 Trades:  {total}  ({wins}W / {losses}L)",
        ]

        if wins:
            avg_win = sum(tr.pnl_usd for tr in t.completed if "win" in tr.reason) / wins
            lines.append(f"  ✅ Avg win:  ${avg_win:+.2f}")
        if losses:
            avg_loss = sum(tr.pnl_usd for tr in t.completed if "loss" in tr.reason) / losses
            lines.append(f"  ❌ Avg loss: ${avg_loss:+.2f}")
        if total:
            avg_conf = sum(tr.confidence for tr in t.completed) / total
            lines.append(f"  🧠 Avg conf: {avg_conf:.0%}")

        lines.extend([
            "",
            f"  ⏱ Uptime: {int(elapsed // 60)}m",
            f"  📡 CLOB updates: {t._clob_updates}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ])
        return "\n".join(lines)

    def _fmt_summary(self) -> str:
        """Final session summary (sent on shutdown)."""
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
            "🛑 <b>SESSION ENDED</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  ⏱ Duration: {int(elapsed // 60)}m\n"
            f"  🏆 {wins}W / {losses}L ({wr:.0f}%)\n"
            f"  💰 PnL: <b>${pnl:+.2f}</b>\n"
            f"  💵 $100 → <b>${t._capital:.2f}</b> ({cap_chg:+.1f}%)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
