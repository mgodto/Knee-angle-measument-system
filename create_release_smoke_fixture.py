from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw


FIXTURE_SIZE = (512, 1024)


def create_fixture(output_path: Path) -> None:
    """Write a deterministic, non-clinical, single-leg-shaped smoke-test PNG."""

    width, height = FIXTURE_SIZE
    image = Image.new("L", FIXTURE_SIZE, color=18)
    draw = ImageDraw.Draw(image)

    # A mild vertical gradient prevents percentile normalization from seeing a
    # degenerate image while keeping this visibly synthetic and non-clinical.
    for y in range(height):
        tone = 18 + (20 * y // (height - 1))
        draw.line((0, y, width - 1, y), fill=tone)

    # Simple geometric anatomy exercises the complete image-to-export path
    # without embedding patient data in release artifacts.  The composition
    # intentionally spans hip, knee, and ankle like a standing long-leg view.
    draw.polygon(
        ((56, 28), (420, 28), (390, 254), (332, 514), (352, 986), (154, 986), (176, 520), (110, 254)),
        fill=48,
    )
    draw.ellipse((56, 44, 314, 278), outline=118, width=13)
    draw.ellipse((110, 92, 238, 224), outline=88, width=10)
    draw.ellipse((188, 164, 282, 258), fill=112, outline=212, width=8)
    draw.polygon(((236, 206), (310, 226), (292, 292), (226, 266)), fill=106, outline=198)

    draw.polygon(((230, 254), (294, 268), (306, 528), (216, 528)), fill=82)
    draw.line((230, 254, 216, 528), fill=204, width=8)
    draw.line((294, 268, 306, 528), fill=204, width=8)
    draw.ellipse((174, 486, 264, 616), fill=112, outline=218, width=8)
    draw.ellipse((252, 486, 342, 616), fill=112, outline=218, width=8)

    draw.rectangle((156, 606, 360, 628), fill=25)
    draw.rounded_rectangle((170, 630, 344, 706), radius=24, fill=104, outline=214, width=8)
    draw.polygon(((214, 684), (302, 684), (288, 936), (226, 936)), fill=76)
    draw.line((214, 684, 226, 936), fill=204, width=7)
    draw.line((302, 684, 288, 936), fill=204, width=7)
    draw.ellipse((194, 910, 316, 986), fill=100, outline=214, width=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=False, compress_level=9)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the deterministic non-clinical image used by release smoke tests."
    )
    parser.add_argument("output", type=Path, help="Destination PNG path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    create_fixture(args.output)


if __name__ == "__main__":
    main()
