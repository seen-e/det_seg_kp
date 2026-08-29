"""Pixel decoder factory."""
from __future__ import annotations

from typing import Any, Mapping, Union

from ..config import PixelDecoderConfig
from .pixel_decoder import PixelDecoder

PixelDecoderSpec = Union[PixelDecoderConfig, Mapping[str, Any]]


def parse_pixel_decoder_cfg(cfg: PixelDecoderSpec) -> PixelDecoderConfig:
    """Normalize dict / dataclass into :class:`PixelDecoderConfig`."""
    if isinstance(cfg, PixelDecoderConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return PixelDecoderConfig(**dict(cfg))
    raise TypeError(
        f"pixel_decoder must be PixelDecoderConfig or dict; got {type(cfg).__name__}"
    )


def build_pixel_decoder(cfg: PixelDecoderSpec) -> PixelDecoder:
    """Build a :class:`PixelDecoder` from config (only argument)."""
    return PixelDecoder(parse_pixel_decoder_cfg(cfg))


__all__ = [
    "PixelDecoder",
    "build_pixel_decoder",
    "parse_pixel_decoder_cfg",
]
