"""Rich Live TUI Dashboard for POLY-DESTROYER.

Beautiful terminal interface showing real-time trading state:
  - Header: mode, SOL price, capital, PnL, uptime
  - Model sets: binance + hermes model status
  - Active markets: progress bars, prices, orderbook
  - Open positions: live P&L tracking
  - Pending signals: queued entries
  - Trade history: recent wins/losses
  - Recording stats: ticks recorded
"""

import time
import asyncio
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.columns import Columns
from rich import box

if TYPE_CHECKING:
    from live_trader.ml_shares_trader import MLSharesTrader


class TradingDashboard:
    """Rich Live dashboard — refreshes every 1s in the terminal."""

    def __init__(self, trader: "MLSharesTrader"):
        self.trader = trader
        self.console = Console()
        self._live: Optional[Live] = None

    # ──────────────────────────────────────────────────────────
    #  MAIN LOOP
    # ──────────────────────────────────────────────────────────

    async def run(self):
        """Async loop: update Live display every 1s."""
        await asyncio.sleep(10.0)  # wait for warmup

        # Suppress console logging — all logs go to file only
        try:
            from core.utils.logger import suppress_console
            suppress_console()
        except Exception:
            pass

        try:
            with Live(console=self.console, refresh_per_second=2, screen=False) as live:
                self._live = live
                _err_count = 0
                while self.trader._running:
                    try:
                        live.update(self._build())
                        _err_count = 0
                    except Exception as e:
                        _err_count += 1
                        if _err_count <= 3:
                            try:
                                from loguru import logger as _log
                                _log.warning(f"  TUI render error #{_err_count}: {e}")
                            except Exception:
                                pass
                        # Show error panel instead of freezing
                        try:
                            live.update(Panel(
                                f"TUI render error: {e}\nBot is still running.",
                                title="[red bold]⚠ TUI Error[/]",
                                border_style="red",
                            ))
                        except Exception:
                            pass
                    await asyncio.sleep(1.0)
        except Exception:
            pass  # fallback to log-based status

    # ──────────────────────────────────────────────────────────
    #  LAYOUT
    # ──────────────────────────────────────────────────────────

    def _build(self) -> Panel:
        """Build the full dashboard layout."""
        t = self.trader
        elapsed = time.time() - t._start_ts

        # Header
        header = self._build_header(elapsed)

        # Markets table
        markets_table = self._build_markets()

        # Positions + Pending side by side
        positions = self._build_positions()
        pending = self._build_pending()

        # Trade history (compact)
        history = self._build_history()

        # Recording stats
        recording = self._build_recording()

        # Compose
        parts = [header, "", markets_table, "", positions, "", pending]
        if history:
            parts.extend(["", history])
        parts.extend(["", recording])

        # Use a group
        from rich.console import Group
        group = Group(*[p for p in parts if p])

        mode = "DRY RUN" if t.dry_run else "LIVE"
        border = "blue" if t.dry_run else "red"
        return Panel(
            group,
            title=f"[bold white] 🤖 POLY-DESTROYER  │  {mode}  │  {int(elapsed//3600)}h{int(elapsed%3600//60):02d}m [/]",
            border_style=border,
            box=box.DOUBLE,
        )

    # ──────────────────────────────────────────────────────────
    #  HEADER
    # ──────────────────────────────────────────────────────────

    def _build_header(self, elapsed: float) -> Text:
        t = self.trader
        wins = sum(1 for tr in t.completed if "win" in tr.reason)
        total = len(t.completed)
        losses = total - wins
        wr = wins / max(total, 1) * 100
        pnl = sum(tr.pnl_usd for tr in t.completed)
        cap_chg = (t._capital / 100 - 1) * 100

        # Model info
        n_models = sum(len(s.get("models", {})) for s in getattr(t, '_model_sets', {}).values())
        sources = list(getattr(t, '_model_sets', {}).keys())
        src_str = " + ".join(sources) if sources else "?"

        txt = Text()
        txt.append("  📈 SOL ", style="dim")
        txt.append(f"${t._sol_price:.2f}", style="bold cyan")
        binance_sol = getattr(t, '_binance_sol', 0)
        if binance_sol > 0:
            diff = t._sol_price - binance_sol
            txt.append(f" (Bin ${binance_sol:.2f} Δ{diff:+.3f})", style="dim")
        txt.append("  │  💵 ", style="dim")
        txt.append(f"${t._capital:.2f}", style="bold green" if cap_chg >= 0 else "bold red")
        txt.append(f" ({cap_chg:+.1f}%)", style="green" if cap_chg >= 0 else "red")
        txt.append("  │  🏆 ", style="dim")
        txt.append(f"{wins}W {losses}L", style="bold white")
        txt.append(f" {wr:.0f}%", style="green" if wr >= 50 else "red")
        txt.append(f"  │  PnL ", style="dim")
        txt.append(f"${pnl:+.2f}", style="bold green" if pnl >= 0 else "bold red")
        txt.append(f"\n  🧠 {n_models} models ({src_str})", style="dim")
        txt.append(f"  │  📡 {t._clob_updates} WS", style="dim")
        txt.append(f"  │  {len(t._active_markets)} markets", style="dim")

        return txt

    # ──────────────────────────────────────────────────────────
    #  ACTIVE MARKETS
    # ──────────────────────────────────────────────────────────

    def _build_markets(self) -> Table:
        t = self.trader
        table = Table(
            title="[bold]Active Markets[/]",
            box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan",
            expand=True, padding=(0, 1),
        )
        table.add_column("Market", style="white", width=28, no_wrap=True)
        table.add_column("Dur", style="dim", width=4, justify="center")
        table.add_column("PTB", style="yellow", width=8, justify="right")
        table.add_column("SOL Δ", width=8, justify="right")
        table.add_column("UP Ask", style="green", width=7, justify="right")
        table.add_column("DN Ask", style="red", width=7, justify="right")
        table.add_column("Sprd", style="dim", width=6, justify="right")
        table.add_column("WS", style="dim", width=2, justify="center")
        table.add_column("Progress", width=20)

        now = time.time()
        sorted_markets = sorted(
            t._active_markets.items(),
            key=lambda x: x[1].get("market").time_remaining_ms if x[1].get("market") else 999999,
        )

        for slug, info in sorted_markets[:15]:  # cap at 15
            market = info.get("market")
            if not market:
                continue
            ptb = info.get("ptb", t._sol_price)
            dur = market.duration_minutes
            delta = t._sol_price - ptb

            # Progress
            total_ms = market.time_remaining_ms + market.time_elapsed_ms
            pct = market.time_elapsed_ms / max(total_ms, 1)
            pbar = self._progress_bar(pct, 14)

            # Time remaining
            secs_left = market.time_remaining_ms / 1000
            if secs_left >= 60:
                tl = f"{int(secs_left//60)}m"
            else:
                tl = f"{secs_left:.0f}s"

            delta_style = "green" if delta > 0 else "red" if delta < 0 else "dim"

            # Real ask prices (what you actually pay)
            up_ask = market.best_ask if market.best_ask > 0 else market.yes_price
            dn_ask = market.no_best_ask if market.no_best_ask > 0 else market.no_price
            ws_ok = "✓" if slug in t._ws_price_confirmed else "·"

            table.add_row(
                slug[-28:],
                f"{dur}m",
                f"${ptb:.2f}",
                Text(f"{delta:+.3f}", style=delta_style),
                f"${up_ask:.3f}",
                f"${dn_ask:.3f}",
                f"${market.spread:.3f}",
                ws_ok,
                Text.assemble(pbar, f" {pct*100:.0f}% {tl}"),
            )

        if not sorted_markets:
            table.add_row("  waiting for markets...", "", "", "", "", "", "", "", "")

        return table

    # ──────────────────────────────────────────────────────────
    #  POSITIONS
    # ──────────────────────────────────────────────────────────

    def _build_positions(self) -> Panel:
        t = self.trader
        if not t.positions:
            return Panel(
                Text("  No open positions", style="dim italic"),
                title=f"[bold]📌 Positions (0/{t.max_positions})[/]",
                border_style="dim", box=box.ROUNDED,
            )

        table = Table(box=None, show_header=True, header_style="bold", expand=True, padding=(0, 1))
        table.add_column("Dir", width=4)
        table.add_column("Market", width=20, no_wrap=True)
        table.add_column("Entry", width=7, justify="right")
        table.add_column("Conf", width=5, justify="right")
        table.add_column("SOL Δ", width=8, justify="right")
        table.add_column("Status", width=6, justify="center")
        table.add_column("Time", width=6, justify="right")

        for pos in t.positions:
            delta = t._sol_price - pos.price_to_beat
            winning = (pos.direction == "UP" and delta > 0) or (pos.direction == "DOWN" and delta < 0)
            icon = "🟢" if winning else "🔴"
            arrow = "▲" if pos.direction == "UP" else "▼"

            secs = 0
            if pos.end_date:
                try:
                    end_ts = __import__('datetime').datetime.fromisoformat(
                        pos.end_date.replace("Z", "+00:00")).timestamp()
                    secs = max(0, end_ts - time.time())
                except (ValueError, TypeError):
                    pass
            tl = f"{int(secs//60)}m" if secs >= 60 else f"{secs:.0f}s"

            table.add_row(
                Text(f"{arrow}", style="green" if pos.direction == "UP" else "red"),
                pos.market_slug[-20:],
                f"${pos.entry_price:.3f}",
                f"{pos.confidence:.0%}",
                Text(f"{delta:+.3f}", style="green" if winning else "red"),
                icon,
                tl,
            )

        return Panel(
            table,
            title=f"[bold]📌 Positions ({len(t.positions)}/{t.max_positions})[/]",
            border_style="green" if any(
                (p.direction == "UP" and t._sol_price > p.price_to_beat) or
                (p.direction == "DOWN" and t._sol_price < p.price_to_beat)
                for p in t.positions
            ) else "yellow",
            box=box.ROUNDED,
        )

    # ──────────────────────────────────────────────────────────
    #  PENDING
    # ──────────────────────────────────────────────────────────

    def _build_pending(self) -> Panel:
        t = self.trader
        if not t._pending_signals:
            return Panel(
                Text("  No pending signals", style="dim italic"),
                title="[bold]⏳ Pending (0)[/]",
                border_style="dim", box=box.ROUNDED,
            )

        lines = []
        for slug, sig in list(t._pending_signals.items())[:5]:
            info = t._active_markets.get(slug, {})
            market = info.get("market")
            if not market:
                continue
            d = sig.get("direction", "?")
            conf = sig.get("conf", 0)
            arrow = "▲" if d == "UP" else "▼"
            cur = market.best_ask if d == "UP" else market.no_best_ask
            if cur <= 0:
                cur = market.yes_price if d == "UP" else market.no_price
            lines.append(
                f"  {arrow} {slug[-20:]}  {d} ${cur:.3f} > ${t.max_share_price:.2f} ({conf:.0%})"
            )

        return Panel(
            "\n".join(lines) if lines else "  ...",
            title=f"[bold]⏳ Pending ({len(t._pending_signals)})[/]",
            border_style="yellow", box=box.ROUNDED,
        )

    # ──────────────────────────────────────────────────────────
    #  HISTORY
    # ──────────────────────────────────────────────────────────

    def _build_history(self) -> Optional[Panel]:
        t = self.trader
        if not t.completed:
            return None

        recent = list(reversed(t.completed[-5:]))
        lines = []
        for tr in recent:
            won = "win" in tr.reason
            icon = "✅" if won else "❌"
            arrow = "▲" if tr.direction == "UP" else "▼"
            slug = getattr(tr, 'market_slug', None) or getattr(tr, 'slug', '?')
            lines.append(
                f"  {icon} {arrow} {slug[-18:]}  "
                f"{tr.confidence:.0%}  {tr.pnl_pct:+.0f}% (${tr.pnl_usd:+.2f})"
            )

        return Panel(
            "\n".join(lines),
            title=f"[bold]📜 Recent Trades ({len(t.completed)} total)[/]",
            border_style="cyan", box=box.ROUNDED,
        )

    # ──────────────────────────────────────────────────────────
    #  RECORDING
    # ──────────────────────────────────────────────────────────

    def _build_recording(self) -> Text:
        t = self.trader
        ticks_ml = getattr(t, '_ticks_recorded', 0)
        rec = getattr(t, '_recorder', None)
        rec_snaps = rec._total_snapshots if rec else 0
        rec_errs = rec._errors if rec else 0
        txt = Text()
        txt.append("  📹 ", style="dim")
        txt.append(f"ML ticks: {ticks_ml}", style="white")
        txt.append(f"  │  Recorder: {rec_snaps:,} snaps", style="cyan")
        if rec_errs:
            txt.append(f"  │  {rec_errs} errs", style="red")

        # Liquidation stats
        liq = getattr(t, '_liq_recorder', None)
        if liq:
            pressure = liq.get_pressure(t._sol_price, radius=2.0)
            txt.append(f"  │  💀 Liqs: {liq._total_events}", style="yellow")
            imb = pressure["imbalance"]
            imb_style = "red" if imb > 0 else "green" if imb < 0 else "dim"
            txt.append(f" (${imb:+,.0f} imb)", style=imb_style)
        return txt

    # ──────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _progress_bar(pct: float, width: int = 14) -> Text:
        """Colored progress bar."""
        pct = max(0, min(1, pct))
        filled = int(pct * width)
        remaining = width - filled

        # Color gradient based on progress
        if pct < 0.2:
            style = "bright_blue"
        elif pct < 0.5:
            style = "cyan"
        elif pct < 0.8:
            style = "yellow"
        else:
            style = "red"

        bar = Text()
        bar.append("█" * filled, style=style)
        bar.append("░" * remaining, style="dim")
        return bar
