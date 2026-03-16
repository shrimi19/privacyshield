"""
image_redactor.py
-----------------
Draws solid black boxes over PII regions on a PIL Image.
Optionally adds white token labels on top of black boxes.
"""

from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont


def redact_regions(
    image: Image.Image,
    regions: list[dict],
    token_map: dict | None = None,
) -> Image.Image:
    """
    Draw black boxes over all given bounding box regions on the image.
    Always draws a label — either [TOKEN_ID] if found in map, or [REDACTED].
    """
    redacted = image.copy()
    draw = ImageDraw.Draw(redacted)

    # Build reverse map: original_value → token_id
    value_to_token = {}
    if token_map:
        value_to_token = {v: k for k, v in token_map.items()}

    # Load font — scale based on image size for readability
    img_w, img_h = image.size
    font_size = max(16, img_w // 80)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    for region in regions:
        bbox = region["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        original_text = region.get("text", "")

        # ── Find token label ──────────────────────────────────────────────────
        # Try exact match first, then partial match
        label = None
        if original_text in value_to_token:
            label = f"[{value_to_token[original_text]}]"
        else:
            # Try partial match — OCR may have slight differences
            for val, tok in value_to_token.items():
                if val and original_text and (
                    val.lower() in original_text.lower() or
                    original_text.lower() in val.lower()
                ):
                    label = f"[{tok}]"
                    break

        # Always show something on the black box
        if label is None:
            label = "[REDACTED]"

        # ── Draw black box ────────────────────────────────────────────────────
        draw.rectangle([x, y, x + w, y + h], fill="black")

        # ── Draw white label on top ───────────────────────────────────────────
        label_x = x + 4
        label_y = y + max(2, (h - font_size) // 2)

        try:
            bbox_text = draw.textbbox((label_x, label_y), label, font=font)
            text_w = bbox_text[2] - bbox_text[0]

            # Extend black box if label is wider than the redacted region
            if text_w > w - 4:
                draw.rectangle(
                    [x, y, x + text_w + 10, y + h],
                    fill="black"
                )

            draw.text(
                (label_x, label_y),
                label,
                fill="white",
                font=font,
            )
        except Exception:
            draw.text((label_x, label_y), label, fill="white")

    return redacted


def redact_full_image(image: Image.Image) -> Image.Image:
    """
    Blur the entire image — used for photos/faces with no text.
    """
    from PIL import ImageFilter
    return image.filter(ImageFilter.GaussianBlur(radius=15))