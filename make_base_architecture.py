# -*- coding: utf-8 -*-
"""绘制「基础架构」双语示意图（纵向布局、大字号，保证在 GitHub 页面里可读）。

用法：python make_base_architecture.py [--out 输出目录]
默认输出到仓库的 assets/ 目录：base-architecture.zh-CN.png 与 base-architecture.png。
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 1100
ROOT = Path(__file__).resolve().parent

FONT_CN = Path("C:/Windows/Fonts/msyh.ttc")
FONT_EN = Path("C:/Windows/Fonts/arial.ttf")
if not FONT_CN.exists():
    FONT_CN = Path("C:/Windows/Fonts/msyh.ttf")
if not FONT_EN.exists():
    FONT_EN = FONT_CN

BG = (255, 255, 255)
INK = (20, 33, 61)
ARROW = (55, 65, 81)
BLUE = (30, 99, 196)
BLUE_FILL = (232, 240, 254)
ORANGE = (217, 119, 6)
ORANGE_FILL = (255, 243, 224)
DENSE_FILL = (255, 251, 230)
DENSE_EDGE = (184, 134, 11)
DENSE_INK = (124, 74, 3)
GREY_FILL = (243, 244, 246)
GREY_EDGE = (75, 85, 99)

TEXT = {
    "zh": {
        "prompt": ["Prompt"],
        "queue": ["异步微批队列"],
        "dense": "稠密区 · 并发有效窗口",
        "acc": ["小显存加速卡", "前段层"],
        "host": ["大内存主机", "后段层与状态"],
        "edge_now": ["当前微批"],
        "edge_next": ["下一微批重叠"],
        "out": ["Decode 与结果流"],
        "note": ["两台设备在稠密区里对相邻微批同时开工，谁都不用干等"],
        "font": FONT_CN,
        "file": "base-architecture.zh-CN.png",
    },
    "en": {
        "prompt": ["Prompt"],
        "queue": ["Async micro-batch queue"],
        "dense": "Dense region · effective concurrency window",
        "acc": ["Small-VRAM", "accelerator", "front layers"],
        "host": ["Large-memory", "host", "rear layers + state"],
        "edge_now": ["current", "micro-batch"],
        "edge_next": ["next micro-batch", "overlaps"],
        "out": ["Decode and result stream"],
        "note": [
            "Both devices work on adjacent micro-batches",
            "at the same time inside the dense region",
        ],
        "font": FONT_EN,
        "file": "base-architecture.png",
    },
}

# 布局（像素）
ROW1 = (80, 200)
PROMPT_BOX = (300, ROW1[0], 640, ROW1[1])
QUEUE_BOX = (760, ROW1[0], 1300, ROW1[1])
DENSE_BOX = (100, 250, 1500, 760)
DENSE_TITLE_Y = (268, 340)
ACC_BOX = (160, 400, 600, 640)
HOST_BOX = (1000, 400, 1440, 640)
EDGE_NOW_Y = 470
EDGE_NEXT_Y = 590
OUT_BOX = (500, 820, 1100, 930)
NOTE_Y = (955, 1075)
PAD = 12


def font(path, size):
    return ImageFont.truetype(str(path), size)


def contains(outer, inner, padding=0):
    return (
        outer[0] + padding <= inner[0]
        and outer[1] + padding <= inner[1]
        and inner[2] <= outer[2] - padding
        and inner[3] <= outer[3] - padding
    )


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def text_block_size(draw, lines, fnt, spacing):
    widths, heights = [], []
    for line in lines:
        l, t, r, b = draw.textbbox((0, 0), line, font=fnt)
        widths.append(r - l)
        heights.append(b - t)
    total_h = sum(heights) + spacing * (len(lines) - 1)
    return max(widths), total_h, heights


def draw_text_block(draw, center, lines, fnt, fill, spacing=10):
    """居中绘制多行文字，返回整体包围盒。"""
    w, h, heights = text_block_size(draw, lines, fnt, spacing)
    cx, cy = center
    y = cy - h / 2
    for line, lh in zip(lines, heights):
        l, t, r, b = draw.textbbox((0, 0), line, font=fnt)
        x = cx - (r - l) / 2
        draw.text((x - l, y - t), line, font=fnt, fill=fill)
        y += lh + spacing
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def rounded(draw, box, fill, outline, width=5, radius=28):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow_head(draw, tip, direction, color, length=30, half=15):
    x, y = tip
    dx, dy = direction
    if dx:  # 水平
        base = x - dx * length
        pts = [(x, y), (base, y - half), (base, y + half)]
    else:  # 竖直
        base = y - dy * length
        pts = [(x, y), (x - half, base), (x + half, base)]
    draw.polygon(pts, fill=color)


def solid_arrow(draw, start, end, color, width=6):
    draw.line([start, end], fill=color, width=width)
    dx = (end[0] > start[0]) - (end[0] < start[0])
    dy = (end[1] > start[1]) - (end[1] < start[1])
    arrow_head(draw, end, (dx, dy), color)


def dashed_arrow_h(draw, start, end, color, width=6, on=26, off=16):
    x0, y = start
    x1 = end[0]
    x = x0
    while x < x1 - 34:
        seg_end = min(x + on, x1 - 34)
        draw.line([(x, y), (seg_end, y)], fill=color, width=width)
        x += on + off
    arrow_head(draw, end, (1, 0), color)


def render(lang):
    t = TEXT[lang]
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    big = font(t["font"], 48)
    mid = font(t["font"], 44)
    small = font(t["font"], 40)
    label = font(t["font"], 38)
    note = font(t["font"], 36)

    boxes = {}

    # 第一行：Prompt → 队列
    rounded(d, PROMPT_BOX, GREY_FILL, GREY_EDGE)
    boxes["prompt"] = draw_text_block(d, box_center(PROMPT_BOX), t["prompt"], mid, INK)
    rounded(d, QUEUE_BOX, GREY_FILL, GREY_EDGE)
    boxes["queue"] = draw_text_block(d, box_center(QUEUE_BOX), t["queue"], mid, INK)
    solid_arrow(d, (PROMPT_BOX[2] + 4, 140), (QUEUE_BOX[0] - 4, 140), ARROW)

    # 队列 → 稠密区
    qx = (QUEUE_BOX[0] + QUEUE_BOX[2]) // 2
    solid_arrow(d, (qx, QUEUE_BOX[3] + 4), (qx, DENSE_BOX[1] - 4), ARROW)

    # 稠密区大框与标题
    rounded(d, DENSE_BOX, DENSE_FILL, DENSE_EDGE, width=7, radius=36)
    title_center = ((DENSE_BOX[0] + DENSE_BOX[2]) // 2, (DENSE_TITLE_Y[0] + DENSE_TITLE_Y[1]) // 2)
    boxes["dense_title"] = draw_text_block(d, title_center, [t["dense"]], big, DENSE_INK)

    # 两个节点
    rounded(d, ACC_BOX, BLUE_FILL, BLUE, width=6)
    acc_lines = t["acc"]
    boxes["acc"] = draw_multi_size(d, box_center(ACC_BOX), acc_lines, big, small, INK)
    rounded(d, HOST_BOX, ORANGE_FILL, ORANGE, width=6)
    boxes["host"] = draw_multi_size(d, box_center(HOST_BOX), t["host"], big, small, INK)

    # 两条边：实线（当前微批）与虚线（下一微批重叠）
    solid_arrow(d, (ACC_BOX[2] + 4, EDGE_NOW_Y), (HOST_BOX[0] - 4, EDGE_NOW_Y), ARROW)
    dashed_arrow_h(d, (ACC_BOX[2] + 4, EDGE_NEXT_Y), (HOST_BOX[0] - 4, EDGE_NEXT_Y), ARROW)
    gap_cx = (ACC_BOX[2] + HOST_BOX[0]) // 2
    w_now, h_now, _ = text_block_size(d, t["edge_now"], label, 6)
    boxes["edge_now"] = draw_text_block(
        d, (gap_cx, EDGE_NOW_Y - 14 - h_now / 2), t["edge_now"], label, INK, spacing=6
    )
    w_next, h_next, _ = text_block_size(d, t["edge_next"], label, 6)
    boxes["edge_next"] = draw_text_block(
        d, (gap_cx, EDGE_NEXT_Y + 16 + h_next / 2), t["edge_next"], label, INK, spacing=6
    )

    # 稠密区 → 结果流
    ox = (OUT_BOX[0] + OUT_BOX[2]) // 2
    solid_arrow(d, (ox, DENSE_BOX[3] + 4), (ox, OUT_BOX[1] - 4), ARROW)
    rounded(d, OUT_BOX, GREY_FILL, GREY_EDGE)
    boxes["out"] = draw_text_block(d, box_center(OUT_BOX), t["out"], mid, INK)

    # 底部说明
    note_center = (W // 2, (NOTE_Y[0] + NOTE_Y[1]) // 2)
    boxes["note"] = draw_text_block(d, note_center, t["note"], note, ARROW, spacing=8)

    # 断言：文字都在各自的框内，标注不压节点，文字块互不重叠
    assert contains(PROMPT_BOX, boxes["prompt"], PAD), "prompt text overflows"
    assert contains(QUEUE_BOX, boxes["queue"], PAD), "queue text overflows"
    assert contains(ACC_BOX, boxes["acc"], PAD), "accelerator text overflows"
    assert contains(HOST_BOX, boxes["host"], PAD), "host text overflows"
    assert contains(OUT_BOX, boxes["out"], PAD), "output text overflows"
    assert contains(DENSE_BOX, boxes["dense_title"], PAD), "dense title overflows"
    assert contains(DENSE_BOX, boxes["edge_now"], PAD), "edge label (now) leaves dense box"
    assert contains(DENSE_BOX, boxes["edge_next"], PAD), "edge label (next) leaves dense box"
    assert contains((0, 0, W, H), boxes["note"], PAD), "note overflows canvas"
    for key in ("edge_now", "edge_next", "dense_title"):
        assert not overlaps(boxes[key], ACC_BOX), f"{key} overlaps accelerator node"
        assert not overlaps(boxes[key], HOST_BOX), f"{key} overlaps host node"
    keys = list(boxes)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            assert not overlaps(boxes[a], boxes[b]), f"text blocks overlap: {a} / {b}"
    # 标注与边线不压：实线在标注下方、虚线在标注上方
    assert boxes["edge_now"][3] <= EDGE_NOW_Y - 6, "edge label (now) touches the solid edge"
    assert boxes["edge_next"][1] >= EDGE_NEXT_Y + 6, "edge label (next) touches the dashed edge"
    return img


def box_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def draw_multi_size(draw, center, lines, big_font, small_font, fill, spacing=12):
    """第一行用大字号，其余行用小字号，整体居中；返回包围盒。"""
    fonts = [big_font] + [small_font] * (len(lines) - 1)
    metrics = []
    for line, fnt in zip(lines, fonts):
        l, t, r, b = draw.textbbox((0, 0), line, font=fnt)
        metrics.append((line, fnt, l, t, r - l, b - t))
    total_h = sum(m[5] for m in metrics) + spacing * (len(lines) - 1)
    max_w = max(m[4] for m in metrics)
    cx, cy = center
    y = cy - total_h / 2
    for line, fnt, l, t, w, h in metrics:
        draw.text((cx - w / 2 - l, y - t), line, font=fnt, fill=fill)
        y += h + spacing
    return (cx - max_w / 2, cy - total_h / 2, cx + max_w / 2, cy + total_h / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "assets"), help="输出目录")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in ("zh", "en"):
        img = render(lang)
        target = out_dir / TEXT[lang]["file"]
        img.save(target, optimize=True)
        print(f"wrote {target} {img.size[0]}x{img.size[1]}")


if __name__ == "__main__":
    main()
