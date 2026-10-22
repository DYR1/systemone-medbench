"""Images in a request's evidence, as in v1: a top-level "images" list of base64 strings or data URIs, and "<image>"
marks in the state that say where each image goes.

* ``place_images(state, n)``: the state as text with the place of every image. A string state (or the text of a JSON
  state) with exactly n "<image>" marks gets an image at each mark, in order; without marks the images come first
  (one image alone, several as "Image 1: ...", "Image 2: ..." on their own lines), then a blank line and the state.
* ``Media``: the decoded images of one request. Each image is resized by the engine's image processor so that both
  sides are multiples of 32 pixels (16-pixel patches, 2 x 2 merged) and it has between `min_pixels` (65,536) and
  `max_pixels` (1,048,576) pixels, keeping its aspect ratio (``smart_resize``, the rule of the processor); it then
  takes one token per 32 x 32 pixels, so a 1024 x 1024 photo is 1,024 tokens. The package computes that size itself,
  for its token counts, and the engines resize with the same bounds.

In a rendered prompt an image is the placeholder of the folder's chat template (``IMAGE_PLACEHOLDER``); the engine
puts the image's tokens there.
"""
from __future__ import annotations

import base64
import binascii
import io
import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

IMAGE_MARK = "<image>"                      # in a string state: where the next image goes
SLOT = "\x00IMAGE\x00"                      # an image's place in a rendered question, before the chat template
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"   # an image in the chat template's output
IMAGE_PAD = "<|image_pad|>"                 # the token the engine replaces by the image's tokens
FACTOR = 32                                 # pixels per image token side: 16-pixel patches, merged 2 x 2
MIN_PIXELS = 65536
MAX_PIXELS = 1 << 20
MAX_IMAGES = 16                             # images in one request (serve --max-images)
_DATA_URI = re.compile(r"^data:(image/[\w.+-]+)?(;[\w=.+-]+)*;base64,", re.I)


def place_images(text: str, n: int) -> str:
    """The state text with the place of n images (module docstring); raises ValueError when the marks do not match."""
    marks = text.count(IMAGE_MARK)
    if marks == n:
        return text.replace(IMAGE_MARK, SLOT)
    if marks:
        raise ValueError(f"the state has {marks} {IMAGE_MARK} marks for {n} images")
    head = SLOT if n == 1 else "\n".join(f"Image {k + 1}: {SLOT}" for k in range(n))
    return head + ("\n\n" + text if text.strip() else "")


def smart_resize(height: int, width: int, factor: int = FACTOR, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """The size the image processor of these models resizes an image to (transformers' Qwen2-VL smart_resize): both
    sides multiples of `factor`, between min_pixels and max_pixels pixels, close to the original aspect ratio."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"an image's sides may differ by a factor of 200 at most, not {height} x {width}")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def decode(value: Any) -> tuple[bytes, str]:
    """A base64 string or a data URI -> (the encoded image bytes, its MIME type from the URI or "")."""
    if not isinstance(value, str):
        raise ValueError(f"an image must be a base64 string or a data URI, not {type(value).__name__}")
    m = _DATA_URI.match(value)
    mime = (m.group(1) or "").lower() if m else ""
    payload = value[m.end():] if m else value
    try:
        return base64.b64decode(re.sub(r"\s+", "", payload), validate=True), mime
    except (binascii.Error, ValueError):
        raise ValueError("an image must be a base64 string or a data URI") from None


@dataclass
class Media:
    """The images of one request: the encoded bytes as sent, the decoded RGB images (PIL), their sizes after the
    processor's resize and their token counts."""
    raw: list = field(default_factory=list)          # (bytes, MIME type)
    images: list = field(default_factory=list)       # PIL images, RGB, original size
    sizes: list = field(default_factory=list)        # (height, width) after the resize
    tokens: list = field(default_factory=list)       # image tokens per image
    min_pixels: int = MIN_PIXELS
    max_pixels: int = MAX_PIXELS
    _uris: list | None = None

    def __len__(self) -> int:
        return len(self.images)

    def data_uris(self) -> list[str]:
        """Every image as a data URI of its original bytes (what an OpenAI-style server is sent)."""
        if self._uris is None:
            self._uris = [f"data:{mime or 'image/png'};base64,{base64.b64encode(raw).decode('ascii')}"
                          for raw, mime in self.raw]
        return self._uris

    @property
    def processor_size(self) -> dict:
        """The resize bounds in the form of the image processor's `size` setting."""
        return {"longest_edge": self.max_pixels, "shortest_edge": self.min_pixels}


def load_media(values: Sequence[Any], min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS,
               max_images: int = MAX_IMAGES, factor: int = FACTOR) -> Media:
    """Decode the images of a request (base64 strings or data URIs); raises ValueError for anything unreadable."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        raise ValueError("this server cannot read images: install Pillow") from None
    if not isinstance(values, (list, tuple)):
        raise ValueError("images must be a list of base64 strings or data URIs")
    if len(values) > max_images:
        raise ValueError(f"at most {max_images} images per request, not {len(values)}")
    media = Media(min_pixels=min_pixels, max_pixels=max_pixels)
    for k, value in enumerate(values):
        raw, mime = decode(value)
        try:
            img = Image.open(io.BytesIO(raw))
            fmt = (img.format or "").lower()
            img = img.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"image {k + 1} is not a readable image ({e})") from None
        h, w = smart_resize(img.height, img.width, factor, min_pixels, max_pixels)
        media.raw.append((raw, mime or (f"image/{fmt}" if fmt else "")))
        media.images.append(img)
        media.sizes.append((h, w))
        media.tokens.append((h // factor) * (w // factor))
    return media


def multimodal_messages(msgs: list[dict]) -> list[dict]:
    """Chat messages whose contents hold SLOT marks -> the same messages with content parts (text and image), the
    form the chat template renders images from."""
    out = []
    for m in msgs:
        content = m["content"]
        if not isinstance(content, str):
            out.append(m)
            continue
        parts = []
        for k, piece in enumerate(content.split(SLOT)):
            if k:
                parts.append({"type": "image"})
            if piece:
                parts.append({"type": "text", "text": piece})
        out.append({**m, "content": parts})
    return out


def split_prompt(prompt: str) -> list[str]:
    """A rendered prompt -> its text between the image placeholders (one more piece than images)."""
    return prompt.split(IMAGE_PLACEHOLDER)


# The chat template the vLLM client sends with an image request: it writes the parts of the already rendered prompt
# back to back and the placeholder for each image, so the engine reads exactly the package's prompt.
PASS_THROUGH_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['content'] is string -%}{{ message['content'] }}{%- else -%}"
    "{%- for part in message['content'] -%}"
    "{%- if part['type'] == 'text' -%}{{ part['text'] }}{%- else -%}" + IMAGE_PLACEHOLDER + "{%- endif -%}"
    "{%- endfor -%}{%- endif -%}{%- endfor -%}")
