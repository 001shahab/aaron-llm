#!/usr/bin/env python3
# Copyright (c) 2026 3S Holding OU. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Author: Prof. Shahab Anbarjafari <shb@3sholding.com>

"""Render the Aaron A4 infographic.

The poster is generated from code rather than drawn by hand, so a claim printed on it
can be corrected in a diff like any other statement about the library. The palette is
taken from the 3S Holding logo: #8CC2FD, #858585, #2C232E and #E6E7E8.

Not part of the package, and outside the two dependency rule: this is a build tool for
a marketing asset, in the same way ruff is a build tool for the source.

    python3 -m venv /tmp/design && /tmp/design/bin/pip install pillow
    /tmp/design/bin/python marketing/make_infographic.py

The script fails if any text or panel lands outside the margins, or if the blocks do
not fit the page, so a layout mistake cannot be committed silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
LOGO = HERE / "logo.png"
OUT = HERE / "aaron-llm-infographic.png"

# A4 at 300 dpi.
W, H = 2480, 3508
MARGIN = 96
CONTENT = W - 2 * MARGIN

# Vertical budget. HEAD_H is the space a section header takes above its block.
H_HEADER = 420
H_NUMBERS = 138
HEAD_H = 82
H_PAIN = 292
H_HOW = 676
H_PROV = 400
H_CODE = 500
H_PRACTICE = 420
H_FOOTER = 212

# Straight from the logo.
INK = (44, 35, 46)
BLUE = (140, 194, 253)
GREY = (133, 133, 133)
MIST = (230, 231, 232)
WHITE = (255, 255, 255)

# Derived, because the logo blue is too light to read as text on paper.
DEEP = (26, 82, 148)
NAVY = (17, 52, 94)
SLATE = (74, 69, 80)
PAPER = (250, 250, 251)
RULE = (215, 217, 221)
TINT = (242, 247, 253)
CODE_BG = (34, 28, 36)
GREEN = (58, 132, 94)
AMBER = (186, 128, 40)
ROSE = (198, 96, 106)

AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"
FACES = {"heavy": 8, "bold": 0, "demi": 2, "medium": 5, "regular": 7}

_cache: dict[tuple[str, int, int], Any] = {}


def font(kind: str, size: int, *, mono: bool = False) -> Any:
    """Load a face, cached. Sizes are pixels at 300 dpi: divide by 4.17 for points."""
    path = MENLO if mono else AVENIR
    index = (0 if kind == "regular" else 1) if mono else FACES[kind]
    key = (path, size, index)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(path, size, index=index)
    return _cache[key]


class Sheet:
    """The page, with helpers that record anything drawn outside the margins."""

    def __init__(self) -> None:
        self.im = Image.new("RGB", (W, H), PAPER)
        self.d = ImageDraw.Draw(self.im)
        self.violations: list[str] = []

    def _check(self, x0: float, x1: float, what: str) -> None:
        if x0 < MARGIN - 2 or x1 > W - MARGIN + 2:
            self.violations.append(f"{what}: x {x0:.0f}..{x1:.0f} outside margins")

    # Primitives ----------------------------------------------------------------

    def width(self, s: str, f: Any) -> float:
        return float(self.d.textlength(s, font=f))

    def text(
        self,
        xy: tuple[float, float],
        s: str,
        f: Any,
        fill: tuple[int, int, int],
        anchor: str = "la",
        spacing: int = 0,
        check: bool = True,
    ) -> float:
        """Draw one run of text and return its measured width."""
        w = self.width(s, f) if "\n" not in s else 0.0
        if check and "\n" not in s:
            x = xy[0]
            left = x - w if anchor[0] == "r" else (x - w / 2 if anchor[0] == "m" else x)
            self._check(left, left + w, f"text {s[:34]!r}")
        self.d.text(xy, s, font=f, fill=fill, anchor=anchor, spacing=spacing)
        return w

    def panel(
        self,
        box: tuple[float, float, float, float],
        *,
        fill: tuple[int, int, int] = WHITE,
        radius: int = 24,
        outline: tuple[int, int, int] | None = RULE,
        width: int = 2,
        shadow: bool = True,
    ) -> None:
        """A rounded card, with a soft shadow so panels separate from the paper."""
        self._check(box[0], box[2], "panel")
        if shadow:
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(layer).rounded_rectangle(
                (box[0] + 3, box[1] + 7, box[2] + 3, box[3] + 11), radius, fill=(44, 35, 46, 30)
            )
            layer = layer.filter(ImageFilter.GaussianBlur(9))
            self.im = Image.alpha_composite(self.im.convert("RGBA"), layer).convert("RGB")
            self.d = ImageDraw.Draw(self.im)
        self.d.rounded_rectangle(box, radius, fill=fill, outline=outline, width=width)

    def wrap(self, s: str, f: Any, width: float) -> list[str]:
        """Greedy wrap on measured width, so no line can overflow its column."""
        lines: list[str] = []
        for para in s.split("\n"):
            line = ""
            for word in para.split():
                trial = f"{line} {word}".strip()
                if self.width(trial, f) <= width or not line:
                    line = trial
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
        return lines

    def para(
        self,
        x: float,
        y: float,
        s: str,
        f: Any,
        fill: tuple[int, int, int],
        width: float,
        leading: int,
        limit: float | None = None,
    ) -> float:
        """Draw wrapped text and return the y below the last line."""
        for line in self.wrap(s, f, width):
            self.text((x, y), line, f, fill)
            y += leading
        if limit is not None and y > limit:
            self.violations.append(f"text overruns its box by {y - limit:.0f}px: {s[:40]!r}")
        return y

    def chip(
        self,
        x: float,
        y: float,
        label: str,
        *,
        f: Any,
        fg: tuple[int, int, int],
        bg: tuple[int, int, int],
        pad: int = 18,
        h: int = 50,
        radius: int = 14,
        outline: tuple[int, int, int] | None = None,
        right: bool = False,
    ) -> float:
        """A pill, anchored left or right at x. Returns its left edge."""
        w = self.width(label, f) + 2 * pad
        x0 = x - w if right else x
        self._check(x0, x0 + w, f"chip {label[:24]!r}")
        self.d.rounded_rectangle((x0, y, x0 + w, y + h), radius, fill=bg, outline=outline, width=2)
        self.text((x0 + w / 2, y + h / 2 - 1), label, f, fg, anchor="mm", check=False)
        return x0

    def rule(self, x0: float, y: float, x1: float, colour: tuple[int, int, int] = RULE) -> None:
        self.d.line((x0, y, x1, y), fill=colour, width=2)

    def chevron(self, x: float, y: float, size: int, colour: tuple[int, int, int]) -> None:
        self.d.polygon([(x, y - size), (x + size * 0.72, y), (x, y + size)], fill=colour)

    def section(self, y: float, number: str, title: str, kicker: str) -> float:
        """A numbered header with a hairline under it. Returns the top of its block."""
        x, r = MARGIN, 27
        self.d.ellipse((x, y, x + 2 * r, y + 2 * r), fill=INK)
        self.text((x + r, y + r), number, font("heavy", 30), BLUE, anchor="mm", check=False)
        self.text((x + 2 * r + 20, y - 3), title, font("heavy", 43), INK)
        used = self.width(title, font("heavy", 43)) + 2 * r + 20
        self.text((x + used + 24, y + 14), kicker, font("medium", 26), GREY)
        self.rule(x, y + 2 * r + 16, W - MARGIN)
        return y + HEAD_H

    def ghost(
        self,
        size: int,
        xy: tuple[int, int],
        *,
        glyph: float,
        disc: float,
        light_only: bool = True,
    ) -> None:
        """The logo as a watermark.

        The flat disc behind the 3S is given less opacity than the letterforms, so the
        mark is recognisable without laying a grey wash over the page. Restricting it to
        light pixels keeps the dark code blocks and bands clean.
        """
        logo = Image.open(LOGO).convert("RGBA").resize((size, size), Image.LANCZOS)
        r, g, b, a = logo.split()

        def inside(channel: Any, mid: int) -> Any:
            return channel.point(lambda v: 255 if mid - 13 <= v <= mid + 13 else 0)

        flat = ImageChops.multiply(
            ImageChops.multiply(inside(r, 230), inside(g, 231)), inside(b, 232)
        )
        logo.putalpha(
            Image.composite(
                a.point(lambda v: int(v * disc)), a.point(lambda v: int(v * glyph)), flat
            )
        )
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        layer.paste(logo, xy, logo)
        ghosted = Image.alpha_composite(self.im.convert("RGBA"), layer).convert("RGB")
        if light_only:
            light = self.im.convert("L").point(lambda v: 255 if v > 190 else 0)
            self.im = Image.composite(ghosted, self.im, light)
        else:
            self.im = ghosted
        self.d = ImageDraw.Draw(self.im)

    def logo_disc(self, x: int, y: int, size: int) -> None:
        """The logo on a white disc, so it sits cleanly on a dark band."""
        disc = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ImageDraw.Draw(disc).ellipse((0, 0, size - 1, size - 1), fill=(*WHITE, 255))
        logo = Image.open(LOGO).convert("RGBA").resize((size, size), Image.LANCZOS)
        disc = Image.alpha_composite(disc, logo)
        self.im.paste(disc, (x, y), disc)
        self.d = ImageDraw.Draw(self.im)

    def band(
        self, y0: int, y1: int, shapes: list[tuple[str, tuple[int, ...], tuple[int, int, int, int]]]
    ) -> None:
        """A dark band with faint geometry, so it is not a flat rectangle."""
        self.d.rectangle((0, y0, W, y1), fill=INK)
        glow = Image.new("RGBA", (W, y1 - y0), (0, 0, 0, 0))
        g = ImageDraw.Draw(glow)
        for kind, coords, colour in shapes:
            if kind == "ellipse":
                g.ellipse(coords, fill=colour)
            else:
                g.polygon([coords[i : i + 2] for i in range(0, len(coords), 2)], fill=colour)
        patch = Image.alpha_composite(self.im.crop((0, y0, W, y1)).convert("RGBA"), glow)
        self.im.paste(patch.convert("RGB"), (0, y0))
        self.d = ImageDraw.Draw(self.im)

    def code(
        self,
        box: tuple[float, float, float, float],
        lines: list[tuple[str, str]],
        *,
        size: int = 24,
        leading: int = 30,
        title: str | None = None,
    ) -> None:
        """A dark code block. Each line is (kind, text), and kind picks the colour."""
        self.panel(box, fill=CODE_BG, radius=16, outline=None, shadow=True)
        x, y = box[0] + 24, box[1] + 18
        if title:
            self.text((x, y), title, font("demi", 23, mono=True), BLUE, check=False)
            self.rule(x, y + 32, box[2] - 24, (78, 68, 82))
            y += 44
        palette = {
            "code": MIST,
            "hi": BLUE,
            "str": (168, 208, 170),
            "note": (150, 144, 154),
            "warn": (234, 186, 112),
        }
        for kind, line in lines:
            f = font("bold" if kind == "hi" else "regular", size, mono=True)
            self.text((x, y), line, f, palette[kind], check=False)
            y += leading
        overflow = y - box[3]
        if overflow > 0:
            self.violations.append(f"code block overflows by {overflow:.0f}px")


# Sections ----------------------------------------------------------------------


def header(s: Sheet) -> float:
    """Dark band: wordmark, the promise, the author, and how to get it."""
    s.band(
        0,
        H_HEADER,
        [
            ("ellipse", (-380, -430, 540, 490), (*BLUE, 24)),
            ("ellipse", (1760, -560, 2960, 340), (*GREY, 20)),
            ("polygon", (1430, H_HEADER, 1810, 0, 1965, 0, 1585, H_HEADER), (*BLUE, 15)),
            ("polygon", (1660, H_HEADER, 2040, 0, 2100, 0, 1720, H_HEADER), (*BLUE, 10)),
        ],
    )
    s.ghost(760, (W - 700, -190), glyph=0.13, disc=0.05, light_only=False)
    x = MARGIN
    s.text((x, 40), "OPEN SOURCE · APACHE 2.0 · PYTHON 3.11+ · v0.1.0", font("demi", 27), BLUE)
    s.text((x, 72), "AARON", font("heavy", 164), WHITE)
    wm = s.width("AARON", font("heavy", 164))
    s.text((x + wm + 16, 128), "-LLM", font("heavy", 86), BLUE)
    s.text((x, 248), "One client for every model — with a policy", font("demi", 44), MIST)
    s.text((x, 298), "and an audit trail.", font("demi", 44), MIST)
    s.rule(x, 358, x + 1020, (88, 78, 92))
    s.text(
        (x, 370),
        "Designed and developed by Prof. Shahab Anbarjafari · 3S Holding OÜ",
        font("demi", 29),
        BLUE,
    )

    claims = [
        "Refused before the socket, not after the invoice.",
        "Hashes on the record, never the transcript.",
        "No proxy to run. Nothing phones home.",
    ]
    for i, claim in enumerate(claims):
        cy = 248 + 46 * i
        s.d.ellipse((1130, cy + 11, 1144, cy + 25), fill=BLUE)
        s.text((1164, cy), claim, font("medium", 27), MIST)

    right = W - MARGIN
    s.logo_disc(right - 192, 30, 192)
    s.chip(
        right,
        240,
        "pip install aaron-llm",
        f=font("demi", 25, mono=True),
        fg=INK,
        bg=BLUE,
        right=True,
    )
    s.chip(
        right,
        300,
        "github.com/001shahab/aaron-llm",
        f=font("demi", 25),
        fg=INK,
        bg=MIST,
        right=True,
    )
    s.chip(
        right,
        360,
        "tested on CPython 3.11 · 3.12 · 3.13",
        f=font("demi", 24),
        fg=MIST,
        bg=INK,
        outline=(98, 88, 102),
        right=True,
    )
    return H_HEADER


def numbers(s: Sheet, y: float) -> float:
    """The strip that makes the size of the thing concrete."""
    s.d.rectangle((0, y, W, y + H_NUMBERS), fill=BLUE)
    items = [
        ("2", "runtime deps:\nhttpx + pydantic"),
        ("5", "providers, one\nreadable file each"),
        ("3.7k", "lines of code,\nfully typed"),
        ("90.7%", "coverage, and no\nnetwork in a test"),
        ("1", "audit record per\ncall, always"),
        ("6", "policy rules, all\nbefore the network"),
        ("0", "telemetry. It\nnever calls home."),
    ]
    step = CONTENT / len(items)
    for i, (big, small) in enumerate(items):
        cx = MARGIN + step * (i + 0.5)
        s.text((cx, y + 8), big, font("heavy", 56), INK, anchor="ma", check=False)
        s.text((cx, y + 68), small, font("demi", 21), NAVY, anchor="ma", spacing=4, check=False)
        if i:
            s.d.line((cx - step / 2, y + 22, cx - step / 2, y + H_NUMBERS - 22), fill=NAVY, width=2)
    return y + H_NUMBERS


def pain(s: Sheet, y: float) -> float:
    """Four failures that are normal today."""
    top = s.section(y + 6, "1", "THE PAIN IT ADDRESSES", "what is broken today")
    s.panel((MARGIN, top, W - MARGIN, top + H_PAIN))
    items = [
        (
            "Five SDKs, five shapes",
            "Every vendor ships its own client, message format and tool-call encoding. Swapping a "
            "model means rewriting glue rather than changing one string, so nobody swaps and the "
            "first choice quietly becomes permanent.",
        ),
        (
            "Nobody can answer “what left the building?”",
            "Which model actually ran, in which jurisdiction, at what cost, for which tenant. The "
            "answer exists for a few milliseconds inside a client object, and is then thrown away "
            "for good.",
        ),
        (
            "Policy lives in a wiki, not in the call path",
            "“Never send personal data to a US endpoint” is a sentence in a document that nothing "
            "enforces at runtime. A control is only real if the call cannot leave without passing "
            "it first.",
        ),
        (
            "The wrapper is bigger than the problem",
            "A general purpose gateway brings dozens of transitive packages and a server to "
            "operate. Reviewing it is a project of its own, and it is never finished because the "
            "dependency tree keeps moving.",
        ),
    ]
    col = (CONTENT - 128) / 2
    for i, (title, body) in enumerate(items):
        cx = MARGIN + 40 + (col + 48) * (i % 2)
        cy = top + 20 + 140 * (i // 2)
        s.d.rounded_rectangle((cx, cy + 4, cx + 7, cy + 122), 4, fill=ROSE)
        s.text((cx + 24, cy - 2), title, font("heavy", 31), INK)
        s.para(cx + 24, cy + 38, body, font("regular", 25), SLATE, col - 34, 30, limit=cy + 130)
    return top + H_PAIN


def how(s: Sheet, y: float) -> float:
    """The centrepiece: the path one call takes, and the gate in the middle of it."""
    top = s.section(y + 6, "2", "HOW IT WORKS", "one call, in order, nothing hidden")
    s.panel((MARGIN, top, W - MARGIN, top + H_HOW))
    x0 = MARGIN + 36
    inner = CONTENT - 72

    s.code(
        (x0, top + 20, x0 + 1150, top + 112),
        [
            ("code", 'reply = client.chat("openai/gpt-4o", "Summarise this contract",'),
            ("code", '                    max_tokens=800, tags={"tenant": "acme"})'),
        ],
        size=25,
        leading=34,
    )
    s.text((x0 + 1190, top + 26), "A model is always provider/model.", font("demi", 26), INK)
    s.para(
        x0 + 1190,
        top + 60,
        "No hidden routing, no default model, no silent upgrade behind your back. Sync or async, "
        "streaming or not, the same eight steps run in the same order.",
        font("regular", 25),
        SLATE,
        inner - 1190 + 36,
        30,
        limit=top + 126,
    )

    steps = [
        ("1", "RESOLVE", "alias to provider/model,\ncycles refused"),
        ("2", "BUILD", "one canonical request,\nmessages normalised"),
        ("3", "POLICY", "six rules, in order,\nbefore any socket"),
        ("4", "BIND KEY", "read at call time,\nmasked in every repr"),
    ]
    bw = (inner - 3 * 46) / 4
    sy = top + 134
    for i, (n, name, note) in enumerate(steps):
        bx = x0 + (bw + 46) * i
        live = n == "3"
        s.panel(
            (bx, sy, bx + bw, sy + 118),
            fill=INK if live else TINT,
            radius=16,
            outline=None if live else (206, 222, 244),
            shadow=False,
        )
        s.text((bx + 18, sy + 12), n, font("heavy", 29), BLUE if live else DEEP, check=False)
        s.text((bx + 50, sy + 14), name, font("heavy", 29), WHITE if live else INK, check=False)
        s.text(
            (bx + 18, sy + 54),
            note,
            font("regular", 22),
            MIST if live else SLATE,
            spacing=6,
            check=False,
        )
        if i < 3:
            s.chevron(bx + bw + 14, sy + 60, 15, GREY)

    gy = sy + 140
    gh = 244
    s.panel(
        (x0, gy, x0 + inner, gy + gh), fill=TINT, radius=18, outline=(196, 217, 244), shadow=False
    )
    gate = "THE POLICY GATE"
    s.text((x0 + 26, gy + 16), gate, font("heavy", 33), INK, check=False)
    s.text(
        (x0 + 26 + s.width(gate, font("heavy", 33)) + 28, gy + 25),
        "a fixed order, so a refusal always names exactly one rule — and none of it "
        "needs a network call",
        font("medium", 25),
        DEEP,
        check=False,
    )
    rules = [
        ("deny", "a deny always beats\nan allow"),
        ("allow", "empty means allow\neverything"),
        ("residency", "an unknown region is\nrefused, not assumed safe"),
        ("capabilities", "tools, vision, docs,\nstreaming, schema"),
        ("max input\ntokens", "estimated before\nanything is sent"),
        ("max $ per\ncall", "worst case, at\nregistry prices"),
        ("redactors", "email, IP, or a\npattern of your own"),
    ]
    rw = (inner - 52 - 6 * 16) / 7
    for i, (name, note) in enumerate(rules):
        rx = x0 + 26 + (rw + 16) * i
        s.panel(
            (rx, gy + 60, rx + rw, gy + 190),
            fill=WHITE,
            radius=12,
            outline=(212, 226, 246),
            shadow=False,
        )
        s.d.rounded_rectangle((rx, gy + 60, rx + rw, gy + 67), 3, fill=GREEN if i == 6 else BLUE)
        s.text((rx + 14, gy + 76), str(i + 1), font("heavy", 24), BLUE, check=False)
        s.text((rx + 40, gy + 74), name, font("heavy", 24), INK, spacing=3, check=False)
        s.text((rx + 14, gy + 122), note, font("regular", 21), SLATE, spacing=5, check=False)
        if i < 6:
            s.chevron(rx + rw + 3, gy + 128, 9, GREY)
    s.text(
        (x0 + 26, gy + 200),
        'Refused? PolicyViolation names the rule and the detail — or on_violation="fallback" walks '
        "your fallback list and re-runs every rule for each candidate.",
        font("medium", 24),
        NAVY,
        check=False,
    )

    ty = gy + gh + 22
    tail = [
        ("5", "PROVIDER", "One readable file per HTTP API. No vendor SDK, ever."),
        ("6", "TRANSPORT", "httpx, retries with jitter, Retry-After honoured."),
        ("7", "NORMALISE", "The same Response, tool_calls and usage every time."),
        ("8", "AUDIT", "One record: success, refusal, failure — even an abandoned stream."),
    ]
    tw = (inner - 3 * 30) / 4
    for i, (n, name, note) in enumerate(tail):
        bx = x0 + (tw + 30) * i
        s.panel((bx, ty, bx + tw, ty + 112), fill=WHITE, radius=14, outline=RULE, shadow=False)
        s.d.rounded_rectangle((bx, ty, bx + 7, ty + 112), 3, fill=GREEN)
        s.text((bx + 22, ty + 12), n, font("heavy", 26), GREEN, check=False)
        s.text((bx + 52, ty + 12), name, font("heavy", 27), INK, check=False)
        s.para(bx + 22, ty + 52, note, font("regular", 23), SLATE, tw - 44, 29)
    return top + H_HOW


def providers(s: Sheet, y: float) -> float:
    """One interface, five providers, and the escape hatch that covers the rest."""
    top = s.section(y + 6, "3", "ONE INTERFACE, FIVE PROVIDERS", "swap a string, not your code")
    s.panel((MARGIN, top, W - MARGIN, top + H_PROV))
    x0 = MARGIN + 34
    inner = CONTENT - 68
    at = [0.0, 0.30, 0.485, 0.60, 0.695, 0.786, 0.872]
    heads = ["MODEL STRING", "ENDPOINT", "STREAM", "VISION", "DOCS", "REGION", "CREDENTIAL"]
    hy = top + 20
    for i, name in enumerate(heads):
        s.text((x0 + inner * at[i], hy), name, font("heavy", 22), GREY, check=False)
    s.rule(x0, hy + 32, x0 + inner)

    rows = [
        ("openai/gpt-4o", "POST /v1/chat/completions", "SSE", "yes", "PDF", "us", "OPENAI_API_KEY"),
        (
            "anthropic/claude-sonnet-4-5",
            "POST /v1/messages",
            "SSE, typed",
            "yes",
            "PDF",
            "us",
            "ANTHROPIC_API_KEY",
        ),
        ("google/gemini-2.5-pro", ":generateContent", "SSE", "yes", "PDF", "us", "GOOGLE_API_KEY"),
        ("ollama/llama3.1", "POST /api/chat", "NDJSON", "model", "no", "local", "none needed"),
        (
            "openai_compat/<model>",
            "your own base_url",
            "SSE",
            "varies",
            "varies",
            "declared",
            "optional",
        ),
    ]
    ry = hy + 44
    for r, row in enumerate(rows):
        if r % 2 == 0:
            s.d.rounded_rectangle((x0 - 12, ry - 7, x0 + inner + 12, ry + 39), 8, fill=TINT)
        local = row[5] == "local"
        s.text((x0, ry), row[0], font("demi", 25, mono=True), GREEN if local else NAVY, check=False)
        s.text(
            (x0 + inner * at[1], ry + 2), row[1], font("regular", 23, mono=True), SLATE, check=False
        )
        s.text((x0 + inner * at[2], ry + 1), row[2], font("demi", 24), DEEP, check=False)
        s.text((x0 + inner * at[3], ry + 1), row[3], font("regular", 24), SLATE, check=False)
        s.text((x0 + inner * at[4], ry + 1), row[4], font("regular", 24), SLATE, check=False)
        s.text(
            (x0 + inner * at[5], ry + 1),
            row[5],
            font("heavy", 24),
            GREEN if local else AMBER,
            check=False,
        )
        s.text(
            (x0 + inner * at[6], ry + 2), row[6], font("regular", 23, mono=True), SLATE, check=False
        )
        ry += 44
    s.rule(x0, ry + 2, x0 + inner)
    s.para(
        x0,
        ry + 14,
        "openai_compat reaches Groq, Mistral, Together, DeepSeek, Fireworks, OpenRouter, vLLM, "
        "LM Studio and llama.cpp through one file and an explicit base_url. provider_options "
        "passes any documented field straight through, so the abstraction never blocks a vendor "
        "feature, and a house provider ships as an entry point rather than a fork.",
        font("regular", 24),
        SLATE,
        inner,
        30,
        limit=top + H_PROV - 8,
    )
    return top + H_PROV


def in_code(s: Sheet, y: float) -> float:
    """Three snippets: the promise, the policy, and the evidence it leaves behind."""
    top = s.section(y + 6, "4", "WHAT IT LOOKS LIKE", "three snippets, nothing elided")
    col = (CONTENT - 2 * 32) / 3
    s.code(
        (MARGIN, top, MARGIN + col, top + H_CODE),
        [
            ("hi", "from aaron import Aaron"),
            ("code", "client = Aaron()"),
            ("code", 'r = client.chat("ollama/llama3.1",'),
            ("str", '      "Summarise the EU AI Act.")'),
            ("code", "print(r.text, r.usage.total_tokens,"),
            ("code", '      f"${r.cost.usd:.4f}")'),
            ("note", "# the same call, any provider:"),
            ("hi", '#   "openai/gpt-4o"'),
            ("hi", '#   "anthropic/claude-sonnet-4-5"'),
            ("hi", '#   "google/gemini-2.5-pro"'),
            ("note", "# structured output, validated"),
            ("note", "# here, not merely requested:"),
            ("code", "client.extract(m, t, schema=Bill)"),
        ],
        title="30 SECONDS IN",
    )
    s.code(
        (MARGIN + col + 32, top, MARGIN + 2 * col + 32, top + H_CODE),
        [
            ("hi", "policy = Policy("),
            ("code", '  allow=["ollama/*", "anthropic/*"],'),
            ("code", '  deny=["*/gpt-3.5*"],'),
            ("hi", '  residency="eu",'),
            ("code", "  max_usd_per_call=0.50,"),
            ("code", "  max_input_tokens=100_000,"),
            ("code", "  redactors=[EmailRedactor()],"),
            ("code", '  fallback=["ollama/llama3.1"],'),
            ("code", '  on_violation="fallback")'),
            ("code", "Aaron(policy=policy).chat(model,"),
            ("str", '    "Reach me at ada@example.com")'),
            ("warn", "# a US endpoint fails residency, so"),
            ("warn", "# this ran on the local model and the"),
            ("warn", "# address never left the process."),
        ],
        title="THE POLICY, INSIDE THE CALL PATH",
    )
    s.code(
        (MARGIN + 2 * col + 64, top, W - MARGIN, top + H_CODE),
        [
            ("note", "# one line of calls.jsonl, trimmed"),
            ("code", '{"model_requested": "ollama/llama3.1",'),
            ("code", ' "provider_region": "local",'),
            ("code", ' "outcome": "ok", "attempts": 1,'),
            ("code", ' "usage": {"input_tokens": 26,'),
            ("code", '           "output_tokens": 298},'),
            ("code", ' "cost": {"usd": 0.0},'),
            ("hi", ' "policy_snapshot": {"residency":'),
            ("hi", '    "eu", "redactors": ["email"]},'),
            ("code", ' "redactions": 1, "latency_ms": 1204,'),
            ("hi", ' "prompt_sha256": "9f2b…",'),
            ("code", ' "tags": {"tenant": "acme"},'),
            ("warn", ' "content": null}'),
        ],
        title="THE EVIDENCE, ONE RECORD PER CALL",
    )
    return top + H_CODE


def practice(s: Sheet, y: float) -> float:
    """Where it earns its place, with the policy each case would actually set."""
    top = s.section(y + 6, "5", "WHERE IT EARNS ITS PLACE", "six workloads, and what each one sets")
    cards = [
        (
            "Clinical triage · hospital",
            "Summarise referrals without a single patient detail crossing a border.",
            'residency="eu" · fallback=["ollama/*"]\nredactors=[Email(), IpAddress()]',
        ),
        (
            "Claims and KYC · bank, insurer",
            "Thousands of small calls, charged back per business unit at month end.",
            "max_usd_per_call=0.25\ntags={tenant, purpose, ticket}",
        ),
        (
            "Contract review · law firm",
            "Question a PDF, then prove which prompt produced which answer.",
            "documents=[contract.pdf]\nprompt_sha256 · response_sha256",
        ),
        (
            "Approved register · public sector",
            "Only models the authority has cleared, reviewed in a pull request.",
            "aaron.policy.yaml, in git\ndeny=[…] · allow=[…] · CI check",
        ),
        (
            "EU AI Act · deployer duties",
            "Show a reviewer the control was in force at the time of the call.",
            "policy_snapshot on every record\nrefusals recorded, not just calls",
        ),
        (
            "Model evaluation · research",
            "One loop across five providers, with tokens and cost per experiment.",
            "for m in models: client.chat(m, p)\nJsonlSink → audit summarise",
        ),
    ]
    col = (CONTENT - 2 * 30) / 3
    ch = (H_PRACTICE - 14) / 2
    for i, (title, body, cfg) in enumerate(cards):
        cx = MARGIN + (col + 30) * (i % 3)
        cy = top + (ch + 14) * (i // 3)
        s.panel((cx, cy, cx + col, cy + ch), radius=16)
        s.d.rounded_rectangle((cx, cy, cx + col, cy + 7), 3, fill=BLUE)
        s.text((cx + 22, cy + 20), title, font("heavy", 28), INK)
        ny = s.para(
            cx + 22, cy + 62, body, font("regular", 24), SLATE, col - 44, 31, limit=cy + 126
        )
        s.d.rounded_rectangle((cx + 16, ny + 4, cx + col - 16, cy + ch - 12), 10, fill=TINT)
        s.text((cx + 28, ny + 14), cfg, font("demi", 20, mono=True), NAVY, spacing=6, check=False)
    return top + H_PRACTICE


def footer(s: Sheet) -> None:
    """Dark band: the honest limits, the links, the licence, the author."""
    top = H - H_FOOTER
    s.band(
        top,
        H,
        [
            ("ellipse", (1880, -240, 2880, 300), (*BLUE, 18)),
            ("polygon", (140, H - top, 320, 0, 380, 0, 200, H - top), (*BLUE, 12)),
        ],
    )
    x = MARGIN
    s.text((x, top + 18), "AND WHAT IT IS NOT", font("heavy", 30), BLUE)
    s.para(
        x,
        top + 56,
        "No proxy, no gateway to operate, no dashboard, no agent framework, no chains, no "
        "memory, no RAG, no embeddings, no database. Need breadth across a hundred providers? "
        "Use LiteLLM. Need a high throughput gateway? Use Bifrost. Aaron is a library inside "
        "your own process, for the case where you have to explain yourself afterwards.",
        font("regular", 25),
        MIST,
        1900,
        32,
        limit=top + 156,
    )
    s.rule(x, top + 158, x + 1900, (86, 76, 90))
    s.text(
        (x, top + 170),
        "github.com/001shahab/aaron-llm   ·   pip install aaron-llm   ·   Apache 2.0   ·   "
        "© 2026 3S Holding OÜ",
        font("demi", 26),
        BLUE,
    )
    s.logo_disc(W - MARGIN - 130, top + 16, 130)
    s.text(
        (W - MARGIN, top + 170),
        "Prof. Shahab Anbarjafari · shb@3sholding.com",
        font("demi", 26),
        MIST,
        anchor="ra",
    )


def main() -> int:
    """Render the poster and report anything outside the margins or off the page."""
    s = Sheet()
    y = header(s)
    y = numbers(s, y)
    y = pain(s, y)
    y = how(s, y)
    y = providers(s, y)
    y = in_code(s, y)
    y = practice(s, y)
    s.ghost(1820, ((W - 1820) // 2, 1160), glyph=0.24, disc=0.09)
    footer(s)

    if y > H - H_FOOTER - 8:
        s.violations.append(f"blocks end at y={y:.0f}, footer starts at {H - H_FOOTER}")
    s.im.save(OUT, "PNG", dpi=(300, 300))
    print(f"wrote {OUT.name}  {W}x{H}px  A4 at 300 dpi")
    print(f"blocks end at y={y:.0f}, footer starts at {H - H_FOOTER}, page is {H}")
    for problem in s.violations:
        print("  PROBLEM:", problem)
    return 1 if s.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
