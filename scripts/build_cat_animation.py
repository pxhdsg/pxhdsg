"""Build an avatar-matched, frame-by-frame cat animation for the profile README."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FRAMES = ASSETS / "cat-frames"
SOURCES = {"sit": ASSETS / "cat-cutout.png", "wave": ASSETS / "cat-wave.png"}

# Pose, tail angle, vertical bounce, eye state, hold time (milliseconds).
STATES = [
    ("sit", -5, 0, "open", 240),
    ("sit", -3, -2, "open", 140),
    ("sit", 0, -4, "open", 130),
    ("sit", 3, -3, "half", 110),
    ("sit", 5, -2, "closed", 110),
    ("sit", 4, 0, "half", 110),
    ("wave", 1, 1, "open", 180),
    ("wave", -2, -3, "open", 130),
    ("wave", -5, -1, "open", 160),
    ("sit", -3, -2, "open", 130),
    ("sit", 0, 0, "open", 170),
    ("sit", -3, 0, "open", 200),
]


def clean_source(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"))
    labels, _ = ndimage.label(alpha > 28)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    main = labels == sizes.argmax()
    cleaned = np.where(main, alpha, 0).astype("uint8")
    image.putalpha(Image.fromarray(cleaned, "L"))
    return image


def split_tail(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    mask = Image.new("L", image.size)
    draw = ImageDraw.Draw(mask)
    draw.polygon(
        [(60, 920), (135, 858), (280, 828), (410, 850),
         (520, 925), (548, 1040), (528, 1195), (120, 1220), (55, 1130)],
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(4))
    alpha = image.getchannel("A")
    tail = image.copy()
    tail.putalpha(Image.fromarray(np.minimum(np.asarray(alpha), np.asarray(mask)).astype("uint8"), "L"))
    body = image.copy()
    body.putalpha(Image.fromarray(
        (np.asarray(alpha, dtype=np.uint16) * (255 - np.asarray(mask, dtype=np.uint16)) // 255).astype("uint8"),
        "L",
    ))
    return body, tail


def draw_blink(image: Image.Image, state: str) -> None:
    if state == "open":
        return
    draw = ImageDraw.Draw(image)
    if state == "half":
        draw.rectangle((779, 389, 875, 445), fill="#deceba")
        draw.rectangle((979, 389, 1053, 435), fill="#e8d9c5")
        return
    draw.rectangle((775, 388, 882, 497), fill="#e2cfb9")
    draw.rectangle((978, 381, 1059, 489), fill="#ebdac6")
    draw.rectangle((786, 451, 871, 466), fill="#584947")
    draw.rectangle((986, 441, 1052, 456), fill="#584947")
    draw.rectangle((786, 451, 871, 457), fill="#8d7168")
    draw.rectangle((986, 441, 1052, 447), fill="#8d7168")


def make_frame(layers: dict, state: tuple) -> Image.Image:
    pose, tail_angle, bob, eyes, _ = state
    body, tail = layers[pose]
    moved_tail = tail.rotate(tail_angle, center=(506, 1006), resample=Image.BICUBIC)
    image = Image.alpha_composite(moved_tail, body)
    draw_blink(image, eyes)
    image = image.crop((45, 50, 1215, 1220)).resize((328, 328), Image.LANCZOS)
    canvas = Image.new("RGBA", (360, 360))
    canvas.alpha_composite(image, (16, 10 + bob))
    return canvas


def gif_palette(frames: list[Image.Image]) -> Image.Image:
    sample = Image.new("RGB", (360 * len(frames), 360), "#050507")
    for index, frame in enumerate(frames):
        sample.paste(frame, (index * 360, 0), frame)
    quantized = sample.quantize(colors=255, method=Image.MEDIANCUT)
    colors = quantized.getpalette()[:765]
    palette = Image.new("P", (1, 1))
    palette.putpalette([255, 0, 255] + colors + [0] * (768 - 3 - len(colors)))
    return palette


def main() -> None:
    FRAMES.mkdir(exist_ok=True)
    layers = {pose: split_tail(clean_source(path)) for pose, path in SOURCES.items()}
    frames = [make_frame(layers, state) for state in STATES]
    for index, frame in enumerate(frames):
        frame.save(FRAMES / f"frame-{index:02d}.png", optimize=True)

    sheet = Image.new("RGB", (360 * 4, 390 * 3), "#101820")
    sheet_draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x = index % 4 * 360
        y = index // 4 * 390
        sheet.paste(frame, (x, y), frame)
        sheet_draw.text((x + 18, y + 352), f"FRAME {index + 1:02d}", fill="#b7d9e4")
    sheet.save(FRAMES / "contact-sheet.png", optimize=True)

    palette = gif_palette(frames)
    indexed = []
    for frame in frames:
        rgb = Image.new("RGB", frame.size, "#050507")
        rgb.paste(frame, mask=frame.getchannel("A"))
        gif_frame = rgb.quantize(palette=palette, dither=Image.NONE)
        gif_frame.paste(0, mask=frame.getchannel("A").point(lambda value: 255 if value < 100 else 0))
        gif_frame.info["transparency"] = 0
        indexed.append(gif_frame)
    indexed[0].save(
        ASSETS / "snow-cat-animated.gif",
        save_all=True,
        append_images=indexed[1:],
        duration=[state[4] for state in STATES],
        loop=0,
        disposal=2,
        transparency=0,
        optimize=False,
    )
    print(f"Created {len(frames)} frames and {ASSETS / 'snow-cat-animated.gif'}")


if __name__ == "__main__":
    main()
