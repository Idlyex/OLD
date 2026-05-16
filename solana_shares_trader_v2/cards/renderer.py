"""
PnL card renderer — exact 1:1 POLYx design via Puppeteer (Node.js).
Falls back to Pillow if Node.js is unavailable.

Primary: cards/render_card.js → Puppeteer screenshot → pixel-perfect POLYx
Fallback: Pure-Python Pillow renderer (close approximation)
"""
import io
import json
import math
import os
import random
import subprocess
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter

log = logging.getLogger(__name__)

RENDER_CARD_JS = Path(__file__).parent / "render_card.js"

# ═══════════════════════════════════════════════════════════
#  SCALE (2x for retina quality, final downscale)
# ═══════════════════════════════════════════════════════════
S = 2  # scale factor

# ═══════════════════════════════════════════════════════════
#  COLORS & THEMES — exact POLYx match
# ═══════════════════════════════════════════════════════════

THEMES = {
    "emerald": {
        "accent": (0, 208, 132),
        "bg1": (10, 18, 16), "bg2": (12, 24, 20), "bg3": (14, 30, 26),
    },
    "cyan": {
        "accent": (0, 229, 255),
        "bg1": (10, 19, 24), "bg2": (12, 24, 30), "bg3": (14, 29, 37),
    },
    "teal": {
        "accent": (45, 212, 191),
        "bg1": (10, 19, 20), "bg2": (12, 24, 26), "bg3": (14, 30, 32),
    },
    "gold": {
        "accent": (255, 184, 77),
        "bg1": (18, 16, 10), "bg2": (24, 20, 12), "bg3": (30, 26, 14),
    },
    "purple": {
        "accent": (167, 139, 250),
        "bg1": (13, 10, 24), "bg2": (17, 14, 30), "bg3": (21, 18, 37),
    },
    "fuchsia": {
        "accent": (232, 121, 249),
        "bg1": (18, 10, 20), "bg2": (24, 12, 26), "bg3": (30, 14, 32),
    },
    "rose": {
        "accent": (255, 77, 106),
        "bg1": (18, 10, 12), "bg2": (24, 12, 16), "bg3": (30, 14, 20),
    },
    "blue": {
        "accent": (96, 165, 250),
        "bg1": (10, 14, 24), "bg2": (12, 18, 30), "bg3": (14, 22, 37),
    },
    "indigo": {
        "accent": (129, 140, 248),
        "bg1": (11, 12, 24), "bg2": (14, 16, 30), "bg3": (17, 20, 37),
    },
}

WHITE = (255, 255, 255)
WHITE90 = (230, 230, 230)
WHITE80 = (204, 204, 204)
WHITE60 = (153, 153, 153)
WHITE35 = (89, 89, 89)
WHITE30 = (77, 77, 77)
WHITE25 = (64, 64, 64)
WHITE20 = (51, 51, 51)
WHITE04 = (10, 10, 10)
ROSE_ACCENT = (255, 77, 106)

# ═══════════════════════════════════════════════════════════
#  FONTS
# ═══════════════════════════════════════════════════════════

FONT_DIR = Path(__file__).parent / "fonts"
_font_cache: Dict[str, ImageFont.FreeTypeFont] = {}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load font with fallback chain (cached)."""
    key = f"{size}_{bold}"
    if key in _font_cache:
        return _font_cache[key]
    px = size * S
    names = [
        "JetBrainsMono-Bold.ttf" if bold else "JetBrainsMono-Regular.ttf",
        "Inter-Bold.ttf" if bold else "Inter-Regular.ttf",
    ]
    for name in names:
        p = FONT_DIR / name
        if p.exists():
            f = ImageFont.truetype(str(p), px)
            _font_cache[key] = f
            return f
    paths = [
        r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for sp in paths:
        if os.path.exists(sp):
            f = ImageFont.truetype(sp, px)
            _font_cache[key] = f
            return f
    return ImageFont.load_default()


def _alpha(color: tuple, a: int) -> tuple:
    """Append alpha to RGB tuple."""
    return color[:3] + (min(max(a, 0), 255),)


# ═══════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════

def _gradient_bg(W: int, H: int, theme: dict) -> Image.Image:
    """155deg diagonal gradient — matches CSS linear-gradient(155deg, bg1, bg2, bg3, bg1)."""
    img = Image.new("RGBA", (W, H))
    draw = ImageDraw.Draw(img)
    bg1, bg2, bg3 = theme["bg1"], theme["bg2"], theme["bg3"]
    for y in range(H):
        t = y / H
        if t < 0.4:
            f = t / 0.4
            r, g, b = [int(bg1[c] + (bg2[c] - bg1[c]) * f) for c in range(3)]
        elif t < 0.7:
            f = (t - 0.4) / 0.3
            r, g, b = [int(bg2[c] + (bg3[c] - bg2[c]) * f) for c in range(3)]
        else:
            f = (t - 0.7) / 0.3
            r, g, b = [int(bg3[c] + (bg1[c] - bg3[c]) * f) for c in range(3)]
        draw.line([(0, y), (W, y)], fill=(r, g, b, 255))
    return img


def _radial_glow(W: int, H: int, accent: tuple, cy: int = 0, intensity: float = 0.08) -> Image.Image:
    """Radial gradient glow at top center — matches POLYx 70% width, 120px height."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx = W // 2
    max_r = int(W * 0.45)
    for r in range(max_r, 0, -2):
        alpha = int(255 * intensity * (1.0 - r / max_r) ** 2.0)
        if alpha < 1:
            continue
        draw.ellipse(
            [cx - int(r * 1.6), cy - r // 2, cx + int(r * 1.6), cy + int(r * 0.7)],
            fill=_alpha(accent, alpha)
        )
    return layer


def _grid_overlay(W: int, H: int, spacing: int = 40) -> Image.Image:
    """Subtle grid — matches POLYx opacity-[0.015] with 40px grid."""
    spacing *= S
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    alpha = 4  # ~0.015 * 255
    for x in range(0, W, spacing):
        draw.line([(x, 0), (x, H)], fill=(255, 255, 255, alpha), width=1)
    for y in range(0, H, spacing):
        draw.line([(0, y), (W, y)], fill=(255, 255, 255, alpha), width=1)
    return layer


def _mini_chart(W: int, H: int, data: List[float], accent: tuple) -> Image.Image:
    """Mini chart with gradient area fill + polyline — exact POLYx MiniChart.jsx style."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if len(data) < 3:
        return layer
    draw = ImageDraw.Draw(layer)

    mn, mx = min(data), max(data)
    rng = mx - mn or 1
    pad = 2 * S

    points = []
    for i, v in enumerate(data):
        x = int(pad + (i / (len(data) - 1)) * (W - 2 * pad))
        y = int(H - pad - ((v - mn) / rng) * (H - 2 * pad - 4 * S))
        points.append((x, y))

    # Area fill — layered polygons for gradient (0.25 at curve → 0.01 at bottom)
    base_pts = [(points[0][0], H)] + points + [(points[-1][0], H)]
    n_layers = 25
    for li in range(n_layers):
        t = li / n_layers
        alpha = int(255 * (0.25 * (1 - t) + 0.01 * t))
        if alpha < 1:
            continue
        shifted = [(x, min(int(y + (H - y) * t), H)) for x, y in base_pts]
        if len(shifted) >= 3:
            draw.polygon(shifted, fill=_alpha(accent, alpha))

    # Main polyline (1.5px → 2*S at 2x)
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=_alpha(accent, 220), width=max(2, int(1.5 * S)))

    # Endpoint glow dot
    ex, ey = points[-1]
    r = 3 * S
    draw.ellipse([ex - r * 2, ey - r * 2, ex + r * 2, ey + r * 2], fill=_alpha(accent, 25))
    draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=accent + (255,))

    # Zero reference line (dashed feel)
    if mn < 0 < mx:
        zy = int(H - pad - ((0 - mn) / rng) * (H - 2 * pad - 4 * S))
        for dx in range(0, W, 10 * S):
            draw.line([(dx, zy), (min(dx + 4 * S, W), zy)], fill=(255, 255, 255, 6), width=1)

    return layer


def _conf_bar(draw: ImageDraw, x: int, y: int, w: int, h: int, pct: float, accent: tuple, agreed: bool):
    """Confidence bar with rounded fill."""
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=(255, 255, 255, 8))
    fw = max(int(w * pct), h)
    fill_color = _alpha(accent, 200 if agreed else 40)
    draw.rounded_rectangle([x, y, x + fw, y + h], radius=h // 2, fill=fill_color)


# ═══════════════════════════════════════════════════════════
#  THEME PICKER — exact POLYx pickTheme logic
# ═══════════════════════════════════════════════════════════

def _pick_theme(pnl_pct: float, won: bool) -> str:
    if not won:
        return "rose"
    if pnl_pct >= 200:
        return "fuchsia"
    if pnl_pct >= 100:
        return "purple"
    if pnl_pct >= 50:
        return "gold"
    if pnl_pct >= 25:
        return "teal"
    if pnl_pct >= 10:
        return "cyan"
    return "emerald"


# ═══════════════════════════════════════════════════════════
#  PUPPETEER RENDERER (primary — exact 1:1 POLYx)
# ═══════════════════════════════════════════════════════════

def _render_via_puppeteer(trade: Dict[str, Any]) -> Optional[bytes]:
    """Render card via Node.js Puppeteer — exact POLYx replica. Returns PNG bytes or None on failure."""
    if not RENDER_CARD_JS.exists():
        return None
    json_path = ""
    png_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as jf:
            json.dump(trade, jf)
            json_path = jf.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as pf:
            png_path = pf.name

        result = subprocess.run(
            ["node", str(RENDER_CARD_JS), json_path, png_path],
            capture_output=True, text=True, timeout=30,
            cwd=str(RENDER_CARD_JS.parent),
        )

        if result.returncode == 0 and os.path.exists(png_path) and os.path.getsize(png_path) > 0:
            with open(png_path, "rb") as f:
                png_bytes = f.read()
            log.info(f"Puppeteer card rendered: {len(png_bytes)} bytes")
            return png_bytes
        else:
            stderr = (result.stderr or "")[:300]
            log.warning(f"Puppeteer render failed (rc={result.returncode}): {stderr}")
            return None
    except FileNotFoundError:
        log.warning("Node.js not found, falling back to Pillow renderer")
        return None
    except subprocess.TimeoutExpired:
        log.warning("Puppeteer render timed out (30s)")
        return None
    except Exception as e:
        log.warning(f"Puppeteer render error: {e}")
        return None
    finally:
        for p in [json_path, png_path]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


def render_trade_card(trade: Dict[str, Any]) -> bytes:
    """Render trade card. Tries Puppeteer (1:1 POLYx) first, falls back to Pillow."""
    png = _render_via_puppeteer(trade)
    if png:
        return png
    log.info("Using Pillow fallback renderer")
    return _render_trade_card_pillow(trade)


# ═══════════════════════════════════════════════════════════
#  PILLOW FALLBACK RENDERER
# ═══════════════════════════════════════════════════════════

def _render_trade_card_pillow(trade: Dict[str, Any]) -> bytes:
    """Pillow fallback — close approximation of POLYx card. Returns PNG bytes."""
    won = trade.get("won", False)
    pnl_pct = trade.get("pnl_pct", 0)
    pnl_usd = trade.get("pnl_usd", 0)
    direction = trade.get("direction", "?")
    is_up = direction == "UP"

    theme_key = _pick_theme(pnl_pct, won)
    theme = THEMES[theme_key]
    accent = theme["accent"]

    # Canvas 480×620 at 2x (matches POLYx width: 480)
    W, H = 480 * S, 620 * S
    img = _gradient_bg(W, H, theme)

    # Radial glow at top
    glow = _radial_glow(W, H, accent, cy=0, intensity=0.08)
    img = Image.alpha_composite(img, glow)

    # Grid overlay
    grid = _grid_overlay(W, H, spacing=40)
    img = Image.alpha_composite(img, grid)

    draw = ImageDraw.Draw(img)

    # Border with accent glow
    draw.rounded_rectangle(
        [S, S, W - S - 1, H - S - 1], radius=22 * S,
        outline=_alpha(accent, 30), width=S
    )

    # Fonts
    f_8 = _font(8, bold=True)
    f_9 = _font(9, bold=True)
    f_10 = _font(10, bold=True)
    f_10r = _font(10, bold=False)
    f_11 = _font(11, bold=True)
    f_13 = _font(13, bold=True)
    f_13r = _font(13, bold=False)
    f_14 = _font(14, bold=True)
    f_16 = _font(16, bold=True)
    f_hero = _font(52, bold=True)
    f_15 = _font(15, bold=True)
    f_mono = _font(12, bold=False)
    f_mono_b = _font(12, bold=True)

    px = 28 * S  # matches POLYx px-7 ≈ 28px
    y = 24 * S   # matches POLYx pt-6

    # ═══ HEADER ROW ═══
    # Left: SOL icon box + POLYx brand
    icon_sz = 36 * S
    icon_bg = _alpha(accent, 20)
    icon_border = _alpha(accent, 38)
    draw.rounded_rectangle([px, y, px + icon_sz, y + icon_sz], radius=12 * S,
                           fill=icon_bg, outline=icon_border, width=S)
    # SOL "S" inside icon box
    draw.text((px + 9 * S, y + 7 * S), "S", fill=accent, font=f_16)

    # Brand name + date
    draw.text((px + icon_sz + 12 * S, y + 2 * S), "DESTROYER", fill=WHITE90, font=f_14)
    slug = trade.get("slug", "unknown")
    dur_str = "5min" if "5m" in slug else "15min" if "15m" in slug else "?"
    exit_time = trade.get("exit_time", "")
    draw.text((px + icon_sz + 12 * S, y + 20 * S), f"{exit_time}", fill=WHITE25, font=f_10r)

    # Right side: duration tag + shares badge + direction badge
    right_x = W - px

    # Direction badge
    badge_text = f"SOL {direction}"
    bw = 88 * S
    bh = 26 * S
    bx = right_x - bw
    draw.rounded_rectangle([bx, y + 4 * S, bx + bw, y + 4 * S + bh], radius=8 * S,
                           fill=_alpha(accent, 15), outline=_alpha(accent, 50), width=S)
    arrow = "\u25B2" if is_up else "\u25BC"
    draw.text((bx + 12 * S, y + 9 * S), f"{arrow} {badge_text}", fill=accent, font=f_11)

    # Duration mini tag
    dur_w = draw.textlength(dur_str, font=f_10r) + 8 * S
    dur_x = bx - dur_w - 6 * S
    draw.rounded_rectangle([int(dur_x), y + 8 * S, int(dur_x + dur_w), y + 8 * S + 18 * S],
                           radius=4 * S, fill=(255, 255, 255, 8))
    draw.text((int(dur_x) + 4 * S, y + 10 * S), dur_str, fill=WHITE25, font=f_10r)

    y += 48 * S

    # ═══ ROI HERO ═══
    draw.text((px, y), "ROI", fill=WHITE25, font=f_10)
    y += 18 * S

    # Big PnL number with text-shadow glow
    pnl_text = f"{abs(pnl_pct):.0f}%" if abs(pnl_pct) >= 100 else f"{abs(pnl_pct):.1f}%"

    # Glow behind text (POLYx: 0 0 40px accent 0.3, 0 0 80px accent 0.15)
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            glow_draw.text((px + dx * S, y + dy * S), pnl_text, fill=_alpha(accent, 40), font=f_hero)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=8 * S))
    img = Image.alpha_composite(img, glow_layer)
    draw = ImageDraw.Draw(img)

    # Arrow icon
    arrow_icon = "\u2191" if won else "\u2193"
    draw.text((px, y + 8 * S), arrow_icon, fill=_alpha(accent, 200), font=_font(20, bold=True))

    # Main text
    draw.text((px + 28 * S, y), pnl_text, fill=accent, font=f_hero)

    # Dollar + hold time to the right
    hero_text_w = draw.textlength(pnl_text, font=f_hero)
    dollar_x = px + 28 * S + int(hero_text_w) + 12 * S

    pnl_dollar = f"{'+'if pnl_usd >= 0 else ''}{pnl_usd:.2f}$"
    draw.text((dollar_x, y + 10 * S), pnl_dollar, fill=WHITE60, font=f_15)

    hold_s = trade.get("hold_time_s", 0)
    hold_str = f"{int(hold_s // 60)}m {int(hold_s % 60)}s" if hold_s else "\u2014"
    draw.text((dollar_x, y + 30 * S), hold_str, fill=WHITE20, font=f_10r)

    # Win/Loss badge (far right)
    badge = "WIN" if won else "LOSS"
    badge_c = accent if won else ROSE_ACCENT
    badge_w = 52 * S
    badge_x = W - px - badge_w
    draw.rounded_rectangle(
        [badge_x, y + 6 * S, badge_x + badge_w, y + 28 * S],
        radius=6 * S,
        fill=_alpha(badge_c, 25),
        outline=_alpha(badge_c, 50),
        width=S,
    )
    draw.text((badge_x + 8 * S, y + 9 * S), badge, fill=badge_c, font=f_10)

    y += 72 * S

    # ═══ MINI CHART ═══ (matches POLYx my-5 -mx-1, height=56)
    chart_data = _generate_chart_data(trade)
    if len(chart_data) > 3:
        chart_w = W - 2 * (px - 4 * S)
        chart_h = 56 * S
        chart_img = _mini_chart(chart_w, chart_h, chart_data, accent)
        img.paste(chart_img, (px - 4 * S, y), chart_img)
        draw = ImageDraw.Draw(img)
        y += chart_h + 20 * S

    # ═══ SOL PRICE MOVEMENT ROW ═══ (matches POLYx rounded-xl with gradient bg)
    sol_entry = trade.get("sol_at_entry", 0)
    sol_exit = trade.get("sol_at_exit", 0)
    ptb = trade.get("ptb", 0)
    sol_delta = sol_exit - sol_entry

    row_h = 56 * S
    row_bg = _alpha(accent, 8)
    row_border = _alpha(accent, 15)
    rx = px - 4 * S
    rw = W - 2 * rx
    draw.rounded_rectangle([rx, y, rx + rw, y + row_h], radius=12 * S,
                           fill=row_bg, outline=row_border, width=S)

    col_w = rw // 3
    labels = [
        ("ENTRY SOL", f"${sol_entry:.2f}", WHITE80),
        ("PTB (OPEN)", f"${ptb:.2f}", WHITE60),
        ("RESOLVE SOL", f"${sol_exit:.2f}", accent if sol_delta >= 0 else ROSE_ACCENT),
    ]
    for i, (label, val, color) in enumerate(labels):
        cx = rx + col_w * i + col_w // 2
        lw = draw.textlength(label, font=_font(9, bold=True))
        vw = draw.textlength(val, font=f_16)
        draw.text((cx - int(lw) // 2, y + 10 * S), label, fill=WHITE25, font=_font(9, bold=True))
        draw.text((cx - int(vw) // 2, y + 28 * S), val, fill=color, font=f_16)
        # Vertical divider
        if i < 2:
            div_x = rx + col_w * (i + 1)
            draw.line([(div_x, y + 12 * S), (div_x, y + row_h - 12 * S)],
                      fill=_alpha(accent, 38), width=S)

    y += row_h + 14 * S

    # ═══ SHARE ENTRY → EXIT ROW ═══ (matches POLYx Share Entry → Share Exit + Peak)
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", 0)
    confidence = trade.get("confidence", 0)

    # Share Entry
    draw.text((px, y), "SHARE ENTRY", fill=WHITE20, font=_font(8, bold=True))
    draw.text((px, y + 12 * S), f"${entry_price:.2f}", fill=WHITE60, font=f_13)

    # Arrow
    arr_x = px + 100 * S
    draw.text((arr_x, y + 12 * S), "\u2192", fill=WHITE20, font=f_13r)

    # Share Exit
    draw.text((arr_x + 24 * S, y), "SHARE EXIT", fill=WHITE20, font=_font(8, bold=True))
    exit_color = _alpha(accent, 180)
    draw.text((arr_x + 24 * S, y + 12 * S), f"${exit_price:.2f}", fill=exit_color, font=f_13)

    # Confidence (right side)
    conf_text = f"Conf {confidence:.0%}"
    conf_w = draw.textlength(conf_text, font=f_13)
    draw.text((W - px - int(conf_w), y + 8 * S), conf_text, fill=accent, font=f_13)

    y += 32 * S

    # Shares + size detail
    shares = trade.get("shares", 0)
    size_usd = trade.get("size_usd", shares * entry_price)
    draw.text((px, y), f"{shares:.1f} shares \u00D7 ${entry_price:.3f} = ${size_usd:.2f}",
              fill=WHITE30, font=_font(10, bold=False))
    y += 20 * S

    # ═══ MODEL AGREEMENT BARS ═══
    all_probs = trade.get("all_model_probs", {})
    if all_probs:
        draw.line([(px, y), (W - px, y)], fill=_alpha(accent, 12), width=S)
        y += 10 * S
        draw.text((px, y), "MODEL AGREEMENT", fill=WHITE20, font=_font(8, bold=True))
        y += 16 * S

        bar_w = 110 * S
        bar_h = 8 * S
        for model_name, p_up in sorted(all_probs.items()):
            dp = max(p_up, 1 - p_up)
            d = "UP" if p_up > 0.5 else "DN"
            agreed = (d == direction)

            name_color = WHITE60 if agreed else WHITE30
            draw.text((px, y), f"{model_name}", fill=name_color, font=f_mono)
            _conf_bar(draw, px + 95 * S, y + 2 * S, bar_w, bar_h, dp, accent, agreed)
            pct_text = f"{dp:.0%} {d}"
            draw.text((px + 95 * S + bar_w + 8 * S, y), pct_text,
                       fill=accent if agreed else WHITE30, font=f_mono)
            y += 18 * S

    # ═══ FOOTER ═══
    y = H - 32 * S
    draw.line([(px, y), (W - px, y)], fill=_alpha(accent, 10), width=S)
    y += 10 * S
    mode = "LIVE" if not trade.get("dry_run") else "DRY RUN"
    dot_color = (255, 60, 60, 255) if mode == "LIVE" else (96, 165, 250, 255)
    draw.ellipse([px, y + 3 * S, px + 7 * S, y + 10 * S], fill=dot_color)
    draw.text((px + 14 * S, y), f"{mode}  \u00B7  DESTROYER 2.0", fill=WHITE30, font=_font(9, bold=False))

    # Downscale to final size (anti-alias)
    final = img.resize((480, 620), Image.LANCZOS)

    buf = io.BytesIO()
    final.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _generate_chart_data(trade: Dict[str, Any]) -> List[float]:
    """Generate a synthetic PnL curve based on trade outcome."""
    won = trade.get("won", False)
    pnl_pct = trade.get("pnl_pct", 0)
    n = 30  # data points

    random.seed(hash(trade.get("slug", "")) & 0xFFFFFFFF)

    data = [0.0]
    for i in range(1, n):
        t = i / (n - 1)
        target = pnl_pct * t
        noise = random.gauss(0, abs(pnl_pct) * 0.12 + 1.5)
        val = target + noise
        data.append(val)
    data[-1] = pnl_pct
    return data


# ═══════════════════════════════════════════════════════════
#  SESSION SUMMARY CARD
# ═══════════════════════════════════════════════════════════

def render_session_card(stats: Dict[str, Any]) -> bytes:
    """Render session summary card. Returns PNG bytes."""
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    total = wins + losses
    pnl = stats.get("total_pnl", 0)
    wr = stats.get("win_rate", 0)
    capital = stats.get("capital", 0)
    uptime = stats.get("uptime_min", 0)

    is_profit = pnl >= 0
    theme = THEMES["emerald"] if is_profit else THEMES["rose"]
    accent = theme["accent"]

    W, H = 480 * S, 320 * S
    img = _gradient_bg(W, H, theme)
    img = Image.alpha_composite(img, _radial_glow(W, H, accent, cy=0, intensity=0.07))
    img = Image.alpha_composite(img, _grid_overlay(W, H, spacing=40))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([S, S, W - S - 1, H - S - 1], radius=22 * S,
                           outline=_alpha(accent, 25), width=S)

    px = 28 * S
    y = 22 * S

    draw.text((px, y), "SESSION SUMMARY", fill=accent, font=_font(14, bold=True))
    y += 30 * S

    draw.text((px, y), "TOTAL PNL", fill=WHITE20, font=_font(9, bold=True))
    y += 16 * S
    draw.text((px, y), f"{'+'if pnl >= 0 else ''}{pnl:.2f}$", fill=accent, font=_font(40, bold=True))
    y += 60 * S

    col_w = (W - 2 * px) // 4
    stats_data = [
        ("TRADES", str(total)),
        ("WIN RATE", f"{wr:.0f}%"),
        ("WINS", str(wins)),
        ("LOSSES", str(losses)),
    ]
    for i, (label, val) in enumerate(stats_data):
        cx = px + col_w * i
        draw.text((cx, y), label, fill=WHITE20, font=_font(8, bold=True))
        color = accent if label == "WINS" else ROSE_ACCENT if label == "LOSSES" else WHITE90
        draw.text((cx, y + 16 * S), val, fill=color, font=_font(18, bold=True))

    y += 50 * S
    draw.text((px, y), f"Capital: ${capital:.2f}  \u00B7  Uptime: {int(uptime)}m",
              fill=WHITE30, font=_font(11, bold=False))

    y = H - 28 * S
    draw.line([(px, y), (W - px, y)], fill=_alpha(accent, 10), width=S)
    y += 8 * S
    draw.text((px, y), f"DESTROYER 2.0  \u00B7  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
              fill=WHITE20, font=_font(9, bold=False))

    final = img.resize((480, 320), Image.LANCZOS)
    buf = io.BytesIO()
    final.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
