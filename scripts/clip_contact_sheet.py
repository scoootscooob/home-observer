#!/usr/bin/env python3
"""Render evenly spaced frames of a clip with a normalized coordinate grid for choosing seed boxes."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--width", type=int, default=427)
    parser.add_argument("--box", default=None, help="x_min,y_min,x_max,y_max normalized overlay")
    args = parser.parse_args()
    reader = imageio_ffmpeg.read_frames(str(args.video), pix_fmt="rgb24")
    meta = next(reader)
    decoded = list(reader)
    fps, (width, height) = meta["fps"], meta["size"]
    indices = [round(i * (len(decoded) - 1) / (args.frames - 1)) for i in range(args.frames)]
    tile_h = round(args.width * height / width)
    rows = (len(indices) + args.columns - 1) // args.columns
    sheet = Image.new("RGB", (args.columns * args.width, rows * (tile_h + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    box = [float(v) for v in args.box.split(",")] if args.box else None
    for slot, index in enumerate(indices):
        tile = Image.frombytes("RGB", (width, height), decoded[index]).resize((args.width, tile_h))
        tile_draw = ImageDraw.Draw(tile)
        for k in range(1, 10):
            x, y = round(args.width * k / 10), round(tile_h * k / 10)
            tile_draw.line([(x, 0), (x, tile_h)], fill=(255, 255, 0), width=1)
            tile_draw.line([(0, y), (args.width, y)], fill=(255, 255, 0), width=1)
            tile_draw.text((x + 2, 2), f"{k/10:.1f}", fill=(255, 255, 0))
            tile_draw.text((2, y + 2), f"{k/10:.1f}", fill=(255, 255, 0))
        if box:
            tile_draw.rectangle([box[0] * args.width, box[1] * tile_h, box[2] * args.width, box[3] * tile_h],
                                outline=(255, 0, 0), width=3)
        x0, y0 = (slot % args.columns) * args.width, (slot // args.columns) * (tile_h + 18)
        sheet.paste(tile, (x0, y0))
        draw.text((x0 + 4, y0 + tile_h + 2), f"frame {index} t={index / fps:.2f}s", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=90)
    print({"frames": len(decoded), "fps": fps, "size": [width, height], "output": str(args.output)})


if __name__ == "__main__":
    main()
