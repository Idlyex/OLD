"""Rich Live Dashboard — real-time trading console with tables, progress, equity curve.

Shows:
- Position table with live P&L
- Trade history with win/loss colors
- Risk metrics (Sharpe, DD, equity)
- Feature importance (if model fitted)
- Regime indicator
- ASCII equity curve via plotext
"""

import time
import asyncio
from typing import Dict, Optional, Any

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.columns import Columns
from rich import box

from core.utils.logger import log
from config import config

_dash_cfg = config.get("dashboard", {})


class LiveDashboard:
    """Real-time Rich console dashboard for live trading."""

    def __init__(self):
        self.console = Console()
        self._live: Optional[Live] = None
        self._refresh_ms = _dash_cfg.get("refresh_ms", 500)
        self._running = False
        self._max_log = _dash_cfg.get("max_log_lines", 50)

        # State refs (set by live_trader)
        self._execution_engine = None
        self._risk_manager = None
        self._feature_engine = None
        self._model = None
        self._cex_collector = None
        self._log_lines = []

    def bind(
        self,
        execution_engine=None,
        risk_manager=None,
        feature_engine=None,
        model=None,
        cex_collector=None,
    ):
        """Bind components for data access."""
        self._execution_engine = execution_engine
        self._risk_manager = risk_manager
        self._feature_engine = feature_engine
        self._model = model
        self._cex_collector = cex_collector

    def _build_header(self) -> Panel:
        """Build header panel."""
        mode = config.get("mode", "live")
        dry = config.get("dry_run", True)
        mode_text = f"[bold yellow]DRY RUN[/]" if dry else f"[bold green]LIVE[/]"
        return Panel(
            f"[bold cyan]Solana Shares Trader v2[/] │ {mode_text} │ Mode: {mode.upper()}",
            box=box.DOUBLE,
            style="bold blue",
        )

    def _build_positions_table(self) -> Table:
        """Build open positions table."""
        table = Table(
            title="📊 Open Positions",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("Slug", style="cyan", width=20)
        table.add_column("Dir", width=4)
        table.add_column("Entry", justify="right", width=8)
        table.add_column("Current", justify="right", width=8)
        table.add_column("PnL %", justify="right", width=8)
        table.add_column("PnL $", justify="right", width=8)
        table.add_column("Peak", justify="right", width=7)
        table.add_column("Age", justify="right", width=6)
        table.add_column("Conf", justify="right", width=6)

        if self._execution_engine:
            for key, pos in self._execution_engine.positions.items():
                pnl_color = "green" if pos.pnl_pct >= 0 else "red"
                dir_icon = "▲" if pos.direction == "UP" else "▼"
                dir_color = "green" if pos.direction == "UP" else "red"

                table.add_row(
                    pos.slug,
                    f"[{dir_color}]{dir_icon}[/]",
                    f"${pos.entry_price:.4f}",
                    f"${pos.current_price:.4f}",
                    f"[{pnl_color}]{pos.pnl_pct:+.1f}%[/]",
                    f"[{pnl_color}]${pos.pnl_usd:+.2f}[/]",
                    f"{pos.peak_pnl_pct:.0f}%",
                    f"{pos.age_s:.0f}s",
                    f"{pos.confidence:.2f}",
                )

        if not self._execution_engine or not self._execution_engine.positions:
            table.add_row("—", "—", "—", "—", "—", "—", "—", "—", "—")

        return table

    def _build_stats_panel(self) -> Panel:
        """Build trading stats panel."""
        stats = {}
        if self._execution_engine:
            stats = self._execution_engine.get_stats()

        risk = {}
        if self._risk_manager:
            risk = self._risk_manager.get_risk_metrics()

        wr = stats.get("win_rate", 0)
        wr_color = "green" if wr >= 50 else "yellow" if wr >= 40 else "red"

        text = Text()
        text.append(f"Trades: {stats.get('total_trades', 0)} ", style="bold")
        text.append(f"(W:{stats.get('wins', 0)} L:{stats.get('losses', 0)})\n")
        text.append(f"Win Rate: ", style="bold")
        text.append(f"{wr:.1f}%\n", style=wr_color)
        text.append(f"Total PnL: ", style="bold")
        pnl = stats.get("total_pnl", 0)
        text.append(f"${pnl:+.2f}\n", style="green" if pnl >= 0 else "red")
        text.append(f"Open: {stats.get('open_positions', 0)}\n", style="bold")
        text.append(f"\n")
        text.append(f"Sharpe: {risk.get('sharpe_ratio', 0):.2f}\n", style="bold")
        text.append(f"Max DD: {risk.get('max_drawdown_pct', 0):.1f}%\n", style="bold")
        text.append(f"Equity: ${risk.get('current_equity', 0):.2f}\n", style="bold")

        return Panel(text, title="📈 Performance", box=box.ROUNDED, border_style="green")

    def _build_prices_panel(self) -> Panel:
        """Build current prices panel."""
        text = Text()

        if self._cex_collector:
            for sym in config.get("infrastructure", {}).get("binance", {}).get("symbols", []):
                sym_upper = sym.upper()
                mp = self._cex_collector.mark_prices.get(sym_upper, {})
                price = mp.get("mark_price", 0)
                funding = mp.get("funding_rate", 0)
                text.append(f"{sym_upper}: ", style="bold cyan")
                text.append(f"${price:.2f} ", style="white")
                fr_color = "green" if funding >= 0 else "red"
                text.append(f"FR:{funding*100:.4f}%\n", style=fr_color)
        else:
            text.append("No price data", style="dim")

        return Panel(text, title="💰 Prices", box=box.ROUNDED, border_style="cyan")

    def _build_regime_panel(self) -> Panel:
        """Build regime indicator panel."""
        text = Text()

        regimes = ["🟢 LOW VOL", "🟡 MEDIUM", "🟠 HIGH VOL", "🔴 EXTREME"]
        # Default
        text.append("Regime: ", style="bold")
        text.append("UNKNOWN\n", style="dim")
        text.append("Hurst: —\n", style="dim")

        return Panel(text, title="🎯 Regime", box=box.ROUNDED, border_style="yellow")

    def _build_trades_table(self) -> Table:
        """Build recent trades table."""
        table = Table(
            title="📜 Recent Trades",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold blue",
            expand=True,
        )
        table.add_column("Time", width=8)
        table.add_column("Dir", width=4)
        table.add_column("Entry→Exit", width=20)
        table.add_column("PnL", justify="right", width=10)
        table.add_column("Reason", width=15)

        if self._execution_engine:
            for trade in self._execution_engine.trade_history[-10:]:
                pnl_color = "green" if trade["pnl_pct"] > 0 else "red"
                icon = "✅" if trade["pnl_pct"] > 0 else "❌"
                dir_icon = "▲" if trade["direction"] == "UP" else "▼"

                table.add_row(
                    time.strftime("%H:%M:%S", time.localtime(trade["entry_ts"] / 1000)),
                    dir_icon,
                    f"${trade['entry_price']:.4f}→${trade['exit_price']:.4f}",
                    f"[{pnl_color}]{icon} {trade['pnl_pct']:+.1f}%[/]",
                    trade["exit_reason"][:15],
                )

        return table

    def _build_equity_chart(self) -> Panel:
        """Build ASCII equity curve using plotext."""
        try:
            import plotext as plt

            if self._risk_manager and self._risk_manager._equity_curve:
                equity = list(self._risk_manager._equity_curve)[-200:]
                if len(equity) >= 2:
                    plt.clear_figure()
                    plt.plot(equity, marker="braille")
                    plt.title("Equity Curve")
                    plt.theme("dark")
                    plt.plotsize(60, 12)
                    chart = plt.build()
                    return Panel(chart, title="📉 Equity", box=box.ROUNDED, border_style="magenta")
        except ImportError:
            pass

        return Panel("No equity data yet", title="📉 Equity", box=box.ROUNDED, border_style="magenta")

    def _build_layout(self) -> Layout:
        """Build full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=15),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=1),
        )

        layout["left"].split_column(
            Layout(name="positions", size=12),
            Layout(name="trades"),
        )

        layout["right"].split_column(
            Layout(name="stats"),
            Layout(name="prices", size=8),
            Layout(name="regime", size=8),
        )

        # Fill layout
        layout["header"].update(self._build_header())
        layout["positions"].update(self._build_positions_table())
        layout["trades"].update(self._build_trades_table())
        layout["stats"].update(self._build_stats_panel())
        layout["prices"].update(self._build_prices_panel())
        layout["regime"].update(self._build_regime_panel())
        layout["footer"].update(self._build_equity_chart())

        return layout

    async def start(self):
        """Start the live dashboard update loop."""
        self._running = True
        log.info("Dashboard: starting Rich live display")

        with Live(
            self._build_layout(),
            console=self.console,
            refresh_per_second=1000 / self._refresh_ms,
            screen=True,
        ) as live:
            self._live = live
            while self._running:
                try:
                    live.update(self._build_layout())
                except Exception as e:
                    pass  # Don't crash on render errors
                await asyncio.sleep(self._refresh_ms / 1000)

    def stop(self):
        """Stop the dashboard."""
        self._running = False

    def print_backtest_results(self, results: Dict, trades_df=None, equity_df=None):
        """Print formatted backtest results to console."""
        self.console.print("\n")
        self.console.rule("[bold blue]BACKTEST RESULTS[/]")

        # Stats table
        stats_table = Table(box=box.ROUNDED, show_header=False, expand=True)
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", justify="right")

        wr = results.get("win_rate", 0)
        pnl = results.get("total_pnl_usd", 0)

        stats_table.add_row("Total Trades", str(results.get("total_trades", 0)))
        stats_table.add_row("Win Rate", f"[{'green' if wr >= 50 else 'red'}]{wr:.1f}%[/]")
        stats_table.add_row("Total PnL", f"[{'green' if pnl >= 0 else 'red'}]${pnl:+.2f}[/]")
        stats_table.add_row("Avg Win", f"{results.get('avg_win_pct', 0):+.2f}%")
        stats_table.add_row("Avg Loss", f"{results.get('avg_loss_pct', 0):+.2f}%")
        stats_table.add_row("Max Drawdown", f"{results.get('max_drawdown_pct', 0):.1f}%")
        stats_table.add_row("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
        stats_table.add_row("Profit Factor", f"{results.get('profit_factor', 0):.2f}")
        stats_table.add_row("Avg Hold Time", f"{results.get('avg_hold_time_s', 0):.0f}s")
        stats_table.add_row("Avg Slippage", f"{results.get('avg_slippage_bps', 0):.2f} bps")
        stats_table.add_row(
            "Capital",
            f"${results.get('initial_capital', 0):.2f} → ${results.get('final_equity', 0):.2f}",
        )

        self.console.print(stats_table)

        # Equity curve
        if equity_df is not None and len(equity_df) > 10:
            try:
                import plotext as plt

                plt.clear_figure()
                plt.plot(equity_df["equity"].values, marker="braille")
                plt.title("Equity Curve")
                plt.theme("dark")
                plt.plotsize(80, 20)
                plt.show()
            except ImportError:
                pass

        self.console.print("\n")
