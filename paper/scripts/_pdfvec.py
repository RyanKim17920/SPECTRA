"""Read back the vector geometry of a single-page matplotlib PDF.

Used ONLY to recover data points whose raw inputs were deleted after a paper figure was
made (see extract_ri_vs_step.py / extract_plism_traj.py).  matplotlib's PDF backend writes
path vertices with 6 decimals and marker offsets (``cm /Mk Do``) with 10 decimals, in
points, so with a per-axes data->page transform calibrated from the gridlines the data can
be read back to ~1e-7 in data units -- far below anything a reader of the figure can see.

The parser walks the raw content stream (not PyMuPDF's float32 ``get_drawings``) and
tracks only what matplotlib emits: ``q/Q``, translate-only ``cm``, ``re W n`` clips,
``m/l/c/h`` paths with ``S/f/B`` paint ops, ``RG/rg/G/g`` colours, ``w`` and ``Do``.
PyMuPDF (``fitz``) is used to open the file and to read tick-label text positions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOK = re.compile(rb"\((?:\\.|[^\\()])*\)|<[0-9A-Fa-f\s]*>|\[|\]|/[^\s/\[\]()<>]+|"
                  rb"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[A-Za-z'\"*]+")


@dataclass
class Path_:
    pts: list            # [(x, y), ...] in PDF user space (origin bottom-left)
    op: str              # paint operator: S, f, B, ...
    stroke: tuple
    fill: tuple
    width: float
    clip: tuple | None   # (x, y, w, h) of the active clip rect, rounded to 3 dp


@dataclass
class Mark:
    name: str
    x: float
    y: float
    stroke: tuple
    fill: tuple
    clip: tuple | None
    group: int           # index of the enclosing q..Q block (one Line2D's markers)


@dataclass
class Page:
    width: float
    height: float
    items: list = field(default_factory=list)   # Path_ and Mark in stream order
    words: list = field(default_factory=list)   # fitz words, (x0, y0, x1, y1, text) TOP-left origin


def _num(b):
    return float(b)


def read(pdf_path) -> Page:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    pg = doc[0]
    stream = pg.read_contents()
    page = Page(width=pg.rect.width, height=pg.rect.height,
                words=[tuple(w[:5]) for w in pg.get_text("words")])

    st = {"tx": 0.0, "ty": 0.0, "clip": None, "stroke": (0, 0, 0), "fill": (0, 0, 0), "w": 1.0}
    stack, ops, path, last_rect = [], [], [], None
    arr = None
    group, group_ctr = -1, 0
    for m in _TOK.finditer(stream):
        t = m.group(0)
        if t == b"[":
            arr = []
            continue
        if t == b"]":
            ops.append(arr)
            arr = None
            continue
        if arr is not None:
            arr.append(t)
            continue
        c = t[:1]
        if c in b"(<" or c == b"/" or c.isdigit() or c in b"-+.":
            ops.append(t)
            continue
        op = t.decode()
        if op == "q":
            stack.append((dict(st), group))
            group_ctr += 1
            group = group_ctr
        elif op == "Q":
            st, group = stack.pop()
        elif op == "cm":
            a, b, c_, d, e, f = map(_num, ops[-6:])
            st["tx"] += e
            st["ty"] += f
        elif op == "re":
            last_rect = tuple(map(_num, ops[-4:]))
            path.append(("re", last_rect))
        elif op == "W":
            st["clip"] = tuple(round(v, 3) for v in last_rect)
        elif op == "n":
            path = []
        elif op == "m" or op == "l":
            path.append((_num(ops[-2]) + st["tx"], _num(ops[-1]) + st["ty"]))
        elif op == "c":
            path.append((_num(ops[-2]) + st["tx"], _num(ops[-1]) + st["ty"]))
        elif op in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*"):
            pts = [p for p in path if p and p[0] != "re"]
            if pts:
                page.items.append(Path_(pts, op, st["stroke"], st["fill"], st["w"], st["clip"]))
            path = []
        elif op == "RG":
            st["stroke"] = tuple(map(_num, ops[-3:]))
        elif op == "rg":
            st["fill"] = tuple(map(_num, ops[-3:]))
        elif op == "G":
            st["stroke"] = (_num(ops[-1]),) * 3
        elif op == "g":
            st["fill"] = (_num(ops[-1]),) * 3
        elif op == "w":
            st["w"] = _num(ops[-1])
        elif op == "Do":
            page.items.append(Mark(ops[-1].decode(), st["tx"], st["ty"], st["stroke"],
                                   st["fill"], st["clip"], group))
        ops = []
    return page


def close(c, ref, tol=0.003):
    return c is not None and all(abs(a - b) < tol for a, b in zip(c, ref))


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def axes_frames(page: Page, grid_rgb):
    """Group gridlines by clip rect -> {clip: {"vx": [x...], "hy": [y...]}} (PDF coords)."""
    frames = {}
    for it in page.items:
        if isinstance(it, Path_) and it.op == "S" and close(it.stroke, grid_rgb) \
                and len(it.pts) == 2 and it.clip is not None:
            (x0, y0), (x1, y1) = it.pts
            fr = frames.setdefault(it.clip, {"vx": [], "hy": []})
            if abs(x0 - x1) < 1e-9:
                fr["vx"].append(x0)
            elif abs(y0 - y1) < 1e-9:
                fr["hy"].append(y0)
    return frames


def fit_line(src, dst):
    """Least-squares dst = a*src + b; returns (a, b, max residual)."""
    n = len(src)
    mx, my = sum(src) / n, sum(dst) / n
    sxx = sum((x - mx) ** 2 for x in src)
    a = sum((x - mx) * (y - my) for x, y in zip(src, dst)) / sxx
    b = my - a * mx
    return a, b, max(abs(a * x + b - y) for x, y in zip(src, dst))


def tick_labels(page: Page, clip, side, numeric=True):
    """Tick labels just left of (side='y') or just below (side='x') a clip rect.

    Returns [(value, centre)] with the centre converted to PDF (bottom-left) coordinates.
    """
    x, y, w, h = clip
    top, bot = page.height - (y + h), page.height - y        # clip in top-left coords
    out = []
    for x0, y0, x1, y1, txt in page.words:
        try:
            v = float(txt.replace("−", "-"))
        except ValueError:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if side == "y" and x - 30 <= x1 <= x + 0.5 and top - 6 < cy < bot + 6:
            out.append((v, page.height - cy))
        elif side == "x" and x - 6 < cx < x + w + 6 and bot - 0.5 <= y0 <= bot + 15:
            out.append((v, cx))
    return out


def title_above(page: Page, clip, gap=20):
    x, y, w, h = clip
    top = page.height - (y + h)
    ws = [wd for wd in page.words if x <= (wd[0] + wd[2]) / 2 <= x + w and top - gap < wd[3] <= top + 1]
    return " ".join(wd[4] for wd in sorted(ws, key=lambda q: q[0]))


def calibrate(page: Page, clip, frame, xticks=None):
    """Data->PDF affine map per axis from gridlines matched to tick labels.

    Returns (fx, fy, resid) where fx/fy map PDF coords back to data, and resid is the max
    gridline-fit residual in points (0 up to float noise for a linear axis).
    """
    def pair(lines, labels):
        prs = []
        for v, c in labels:
            g = min(lines, key=lambda q: abs(q - c))
            if abs(g - c) < 1.5:
                prs.append((v, g))
        return prs

    if xticks is not None:
        xs = sorted(frame["vx"])
        assert len(xs) == len(xticks), (xs, xticks)
        px = list(zip(xticks, xs))
    else:
        px = pair(frame["vx"], tick_labels(page, clip, "x"))
    py = pair(frame["hy"], tick_labels(page, clip, "y"))
    assert len(px) >= 2 and len(py) >= 2, (clip, px, py)
    ax, bx, rx = fit_line([p[0] for p in px], [p[1] for p in px])
    ay, by, ry = fit_line([p[0] for p in py], [p[1] for p in py])
    return (lambda X: (X - bx) / ax), (lambda Y: (Y - by) / ay), max(rx, ry)
