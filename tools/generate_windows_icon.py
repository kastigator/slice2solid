from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def _render(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = max(1, int(size * 0.08))
    radius = max(2, int(size * 0.22))

    # Background: deep-blue rounded square.
    bg0 = (10, 32, 60, 255)  # #0A203C
    bg1 = (14, 64, 115, 255)  # #0E4073
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        fill=bg0,
    )

    # Subtle diagonal highlight (simple alpha mask).
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    o = ImageDraw.Draw(overlay)
    o.polygon(
        [
            (margin, margin),
            (size - margin, margin),
            (size - margin, int(size * 0.46)),
            (int(size * 0.46), size - margin),
            (margin, size - margin),
        ],
        fill=(bg1[0], bg1[1], bg1[2], 120),
    )
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # "Slice" layers (3 stripes), slightly offset to suggest stacking.
    layer_colors = [
        (246, 248, 255, 230),
        (170, 211, 255, 230),
        (94, 174, 255, 230),
    ]
    stripe_h = max(2, int(size * 0.105))
    stripe_r = max(2, int(size * 0.08))
    x0_base = int(size * 0.20)
    x1_base = int(size * 0.80)
    y_base = int(size * 0.33)
    step = int(size * 0.14)
    x_step = int(size * 0.02)

    for i, c in enumerate(layer_colors):
        x0 = x0_base - i * x_step
        x1 = x1_base + i * x_step
        y0 = y_base + i * step
        y1 = y0 + stripe_h
        draw.rounded_rectangle([x0, y0, x1, y1], radius=stripe_r, fill=c)

    # Thin outline for contrast on small sizes.
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=radius,
        outline=(255, 255, 255, 40),
        width=max(1, int(size * 0.012)),
    )

    return img


def build_ico(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = _render(256)
    base.save(
        out_path,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tools/installer/slice2solid.ico"))
    args = ap.parse_args()
    build_ico(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

