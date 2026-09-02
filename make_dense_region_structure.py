# -*- coding: utf-8 -*-
"""生成清晰、无越界的「稠密加速结构」双语示意图。"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W, H = 1400, 900
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets" / "dense-region-structure.png"

FONT_CN = Path("C:/Windows/Fonts/msyh.ttc")
FONT_EN = Path("C:/Windows/Fonts/arial.ttf")
if not FONT_CN.exists():
    FONT_CN = Path("C:/Windows/Fonts/msyh.ttf")
if not FONT_EN.exists():
    FONT_EN = FONT_CN


def load_font(path, size):
    return ImageFont.truetype(str(path), size)


def contains(outer, inner, padding=0):
    return (
        outer[0] + padding <= inner[0]
        and outer[1] + padding <= inner[1]
        and inner[2] <= outer[2] - padding
        and inner[3] <= outer[3] - padding
    )


def assert_rect(name, rect, outer=(0, 0, W, H), padding=0):
    assert rect[0] < rect[2] and rect[1] < rect[3], f"{name}: invalid rectangle"
    assert contains(outer, rect, padding), f"{name}: outside {outer}: {rect}"


img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)


def text_checked(xy, text, font, fill, anchor="mm", within=(0, 0, W, H), padding=8):
    bbox = draw.textbbox(xy, text, font=font, anchor=anchor)
    assert contains(within, bbox, padding), f"text overflow: {text!r}, {bbox}, {within}"
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)
    return bbox


def down_arrow(cx, top, bottom, color, channel):
    assert_rect("arrow channel", channel)
    shaft_end = bottom - 18
    arrow_box = (cx - 13, top, cx + 13, bottom)
    assert contains(channel, arrow_box), f"arrow outside channel: {arrow_box}"
    draw.line((cx, top, cx, shaft_end), fill=color, width=6)
    draw.polygon(((cx - 13, shaft_end), (cx + 13, shaft_end), (cx, bottom)), fill=color)


# High-contrast palette.
DARK = (26, 38, 61)
GRAY = (87, 98, 116)
LINE = (102, 112, 133)
SPARSE_FILL = (241, 244, 248)
SPARSE_LINE = (96, 108, 128)
DENSE_FILL = (216, 235, 255)
DENSE_LINE = (10, 92, 201)
BLUE = (15, 102, 220)
BLUE_DARK = (7, 66, 151)
HOST_FILL = (255, 246, 224)
HOST_LINE = (208, 128, 20)
ACCEL_FILL = (232, 243, 255)
WHITE = (255, 255, 255)

f_title = load_font(FONT_CN, 46)
f_subtitle = load_font(FONT_EN, 23)
f_hw = load_font(FONT_CN, 28)
f_hw_en = load_font(FONT_EN, 20)
f_cell = load_font(FONT_CN, 25)
f_zone = load_font(FONT_CN, 36)
f_zone_en = load_font(FONT_EN, 23)
f_body = load_font(FONT_CN, 26)
f_body_en = load_font(FONT_EN, 19)
f_compute = load_font(FONT_EN, 28)
f_compute_sub = load_font(FONT_CN, 19)
f_banner = load_font(FONT_CN, 23)
f_banner_en = load_font(FONT_EN, 17)
f_footer = load_font(FONT_CN, 25)
f_footer_en = load_font(FONT_EN, 19)

# Title.
text_checked((W // 2, 52), "稠密加速结构示意", f_title, DARK)
text_checked((W // 2, 101), "DENSE ACCELERATION — CLEAR MEMORY ZONES", f_subtitle, GRAY)

# Two participating devices, kept in separate cards and arrow lanes.
HOST = (110, 140, 610, 248)
ACCEL = (790, 140, 1290, 248)
assert_rect("host card", HOST)
assert_rect("accelerator card", ACCEL)
draw.rounded_rectangle(HOST, radius=18, fill=HOST_FILL, outline=HOST_LINE, width=4)
draw.rounded_rectangle(ACCEL, radius=18, fill=ACCEL_FILL, outline=DENSE_LINE, width=4)
text_checked((360, 177), "大显存主机", f_hw, DARK, within=HOST, padding=14)
text_checked((360, 218), "LARGE-MEMORY HOST", f_hw_en, GRAY, within=HOST, padding=14)
text_checked((1040, 177), "小显存加速卡", f_hw, BLUE_DARK, within=ACCEL, padding=14)
text_checked((1040, 218), "SMALL-VRAM ACCELERATOR", f_hw_en, GRAY, within=ACCEL, padding=14)
draw.ellipse((664, 156, 736, 228), fill=DARK)
text_checked((700, 192), "+", f_title, WHITE, within=(664, 156, 736, 228), padding=8)

down_arrow(360, 258, 292, HOST_LINE, (338, 252, 382, 296))
down_arrow(1040, 258, 292, BLUE, (1018, 252, 1062, 296))

# One shared memory cell with two non-overlapping zones.
CELL = (65, 300, 1335, 700)
SPARSE = (95, 360, 625, 670)
DIVIDER = (638, 375, 652, 655)
DENSE = (665, 360, 1305, 670)
for name, rect in (("memory cell", CELL), ("sparse region", SPARSE), ("divider", DIVIDER), ("dense region", DENSE)):
    assert_rect(name, rect)
assert contains(CELL, SPARSE, 20)
assert contains(CELL, DENSE, 20)
assert SPARSE[2] < DIVIDER[0] < DIVIDER[2] < DENSE[0]

draw.rounded_rectangle(CELL, radius=30, fill=WHITE, outline=DARK, width=5)
text_checked((700, 329), "显存重叠单元  /  OVERLAPPING MEMORY CELL", f_cell, DARK, within=CELL, padding=15)
draw.rounded_rectangle(SPARSE, radius=20, fill=SPARSE_FILL, outline=SPARSE_LINE, width=4)
draw.rounded_rectangle(DENSE, radius=20, fill=DENSE_FILL, outline=DENSE_LINE, width=5)
draw.rounded_rectangle(DIVIDER, radius=7, fill=BLUE_DARK)

# Sparse region: capacity and complete model residency.
text_checked((360, 400), "稀疏区", f_zone, DARK, within=SPARSE, padding=18)
text_checked((360, 438), "SPARSE REGION", f_zone_en, GRAY, within=SPARSE, padding=18)
text_checked((360, 486), "完整模型驻留", f_body, DARK, within=SPARSE, padding=18)
text_checked((360, 519), "FULL MODEL RESIDENCY", f_body_en, GRAY, within=SPARSE, padding=18)

MODEL_A = (165, 548, 555, 590)
MODEL_B = (165, 606, 555, 648)
for name, rect, label in (("model block A", MODEL_A, "MODEL WEIGHTS  A"), ("model block B", MODEL_B, "MODEL WEIGHTS  B")):
    assert contains(SPARSE, rect, 18), f"{name} outside sparse region"
    draw.rounded_rectangle(rect, radius=10, fill=WHITE, outline=SPARSE_LINE, width=3)
    text_checked(((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), label, f_body_en, DARK, within=rect, padding=9)

# Dense region: prefill and decode compute, with all annotations contained.
text_checked((985, 400), "稠密区", f_zone, BLUE_DARK, within=DENSE, padding=18)
text_checked((985, 438), "DENSE REGION", f_zone_en, BLUE_DARK, within=DENSE, padding=18)

PREFILL = (710, 480, 940, 575)
DECODE = (1030, 480, 1260, 575)
for name, rect in (("prefill card", PREFILL), ("decode card", DECODE)):
    assert contains(DENSE, rect, 20), f"{name} outside dense region"
    draw.rounded_rectangle(rect, radius=16, fill=WHITE, outline=BLUE, width=4)
text_checked((825, 520), "PREFILL", f_compute, BLUE_DARK, within=PREFILL, padding=12)
text_checked((825, 548), "预填充计算", f_compute_sub, GRAY, within=PREFILL, padding=12)
text_checked((1145, 520), "DECODE", f_compute, BLUE_DARK, within=DECODE, padding=12)
text_checked((1145, 548), "解码计算", f_compute_sub, GRAY, within=DECODE, padding=12)
text_checked((985, 527), "+", f_title, BLUE_DARK, within=(950, 485, 1020, 570), padding=6)

BANNER = (740, 590, 1230, 654)
assert contains(DENSE, BANNER, 16)
draw.rounded_rectangle(BANNER, radius=14, fill=BLUE_DARK)
text_checked((985, 610), "显著加速 / 无损", f_banner, WHITE, within=BANNER, padding=8)
text_checked((985, 638), "SIGNIFICANT + LOSSLESS", f_banner_en, WHITE, within=BANNER, padding=7)

# Footer stays in its own lane, outside the memory cell.
text_checked((700, 765), "稀疏区存放模型；稠密区执行 Prefill + Decode", f_footer, DARK)
text_checked((700, 810), "MODEL CAPACITY ON THE LEFT · ACCELERATED COMPUTE ON THE RIGHT", f_footer_en, GRAY)

# Built-in geometry and output checks.
assert not (SPARSE[2] > DENSE[0] and SPARSE[0] < DENSE[2])
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUTPUT, "PNG")

with Image.open(OUTPUT) as check:
    assert check.size == (W, H)
    assert check.mode == "RGB"
    assert all(check.getpixel(point) == WHITE for point in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)))

print(f"PATH={OUTPUT}")
print(f"SIZE={W}x{H}")
print("LAYOUT_CHECKS=PASS")
