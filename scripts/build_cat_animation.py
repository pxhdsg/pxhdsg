"""Build an avatar-matched, frame-by-frame cat animation for the profile README."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FRAMES = ASSETS / "cat-frames"
SOURCES = {
    "sit": ASSETS / "cat-cutout.png",
    "stretch": ASSETS / "cat-stretch.png",
    "leap": ASSETS / "cat-leap.png",
    "sleep": ASSETS / "cat-sleep.png",
}

# Pose, horizontal/vertical offset, size, eyes, hold time, tail angle.
STATES = [
    ("sit", 0, 0, 1, "open", 240, -5),
    ("sit", 0, -3, 1, "open", 150, -2),
    ("sit", 0, -4, 1, "half", 90, 2),
    ("sit", 0, -3, 1, "closed", 100, 5),
    ("sit", 0, 0, 1, "open", 160, 2),
    ("stretch", 0, 0, 1, "open", 220, 0),
    ("stretch", 0, 2, 1.02, "open", 140, 0),
    ("stretch", 0, 0, 1.03, "open", 140, 0),
    ("stretch", 0, -2, 1, "open", 180, 0),
    ("leap", -18, 24, 1, "open", 120, 0),
    ("leap", -10, 8, 1, "open", 110, 0),
    ("leap", 0, -12, 1, "open", 110, 0),
    ("leap", 10, -6, 1, "open", 110, 0),
    ("leap", 18, 20, 1, "open", 130, 0),
    ("sit", 0, -2, 1, "open", 190, -2),
    ("sleep", 0, 0, 1, "closed", 250, 0),
    ("sleep", 0, 0, 1.015, "closed", 230, 0),
    ("sleep", 0, 0, 1.025, "closed", 230, 0),
    ("sleep", 0, 0, 1.015, "closed", 230, 0),
    ("sleep", 0, 0, 1, "closed", 260, 0),
    ("sit", 0, 0, 1, "open", 240, -5),
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
    pose, dx, dy, size, eyes, _, tail_angle = state
    if pose == "sit":
        body, tail = layers[pose]
        moved_tail = tail.rotate(tail_angle, center=(506, 1006), resample=Image.BICUBIC)
        image = Image.alpha_composite(moved_tail, body)
        draw_blink(image, eyes)
    else:
        image = layers[pose].copy()
    bounds = image.getchannel("A").getbbox()
    image = image.crop((max(0, bounds[0] - 10), max(0, bounds[1] - 10),
                        min(image.width, bounds[2] + 10), min(image.height, bounds[3] + 10)))
    max_width, max_height = {
        "sit": (326, 322), "stretch": (326, 315),
        "leap": (330, 290), "sleep": (322, 280),
    }[pose]
    ratio = min(max_width / image.width, max_height / image.height) * size
    image = image.resize((round(image.width * ratio), round(image.height * ratio)), Image.LANCZOS)
    canvas = Image.new("RGBA", (360, 360))
    x = (360 - image.width) // 2 + dx
    y = ((360 - image.height) // 2 + dy if pose == "leap" else 340 - image.height + dy)
    canvas.alpha_composite(image, (x, y))
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
    layers = {pose: (split_tail(clean_source(path)) if pose == "sit" else clean_source(path))
              for pose, path in SOURCES.items()}
    frames = [make_frame(layers, state) for state in STATES]
    for index, frame in enumerate(frames):
        frame.save(FRAMES / f"frame-{index:02d}.png", optimize=True)

    columns = 5
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (360 * columns, 390 * rows), "#101820")
    sheet_draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x = index % columns * 360
        y = index // columns * 390
        sheet.paste(frame, (x, y), frame)
        sheet_draw.text((x + 18, y + 352), f"FRAME {index + 1:02d} / {STATES[index][0].upper()}", fill="#b7d9e4")
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
        duration=[state[5] for state in STATES],
        loop=0,
        disposal=2,
        transparency=0,
        optimize=False,
    )
    print(f"Created {len(frames)} frames and {ASSETS / 'snow-cat-animated.gif'}")


if __name__ == "__main__":
    main()
