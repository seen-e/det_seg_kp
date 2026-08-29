"""Pixel decoder factory."""
from __future__ import annotations

from typing import Any, Mapping, Union

import torch.nn as nn

from ..config import PixelDecoderConfig
from .pixel_decoder import FPNPixelDecoder

PixelDecoderSpec = Union[PixelDecoderConfig, Mapping[str, Any], str]

# Registry name → class. Only FPN is wired for now.
_PIXEL_DECODER_TYPES = {
    "fpn": FPNPixelDecoder,
}

_FAMILY_DEFAULTS = {
    "base_pixel_decoder": "fpn",
}


def list_pixel_decoders() -> list[str]:
    return sorted(_PIXEL_DECODER_TYPES)


def parse_pixel_decoder_cfg(cfg: PixelDecoderSpec) -> PixelDecoderConfig:
    """Normalize dict / str / dataclass into :class:`PixelDecoderConfig`."""
    if isinstance(cfg, PixelDecoderConfig):
        return cfg
    if isinstance(cfg, str):
        return PixelDecoderConfig(type=cfg)
    if isinstance(cfg, Mapping):
        if "type" not in cfg:
            raise ValueError("pixel_decoder dict needs 'type'")
        return PixelDecoderConfig(**dict(cfg))
    raise TypeError(
        f"pixel_decoder must be PixelDecoderConfig, dict, or str; got {type(cfg).__name__}"
    )


def resolve_pixel_decoder_type(type_name: str) -> str:
    """Map ``type`` to a registry key (``base_pixel_decoder`` → ``fpn``)."""
    key = type_name.lower().strip()
    if key in _PIXEL_DECODER_TYPES:
        return key
    if key in _FAMILY_DEFAULTS:
        return _FAMILY_DEFAULTS[key]
    raise ValueError(
        f"Unknown pixel_decoder type '{type_name}'. "
        f"Supported: {', '.join(list_pixel_decoders())}. "
        f"Aliases: {', '.join(sorted(_FAMILY_DEFAULTS))}"
    )


def build_pixel_decoder(cfg: PixelDecoderSpec) -> nn.Module:
    """Build from config. Currently only ``type="fpn"`` is supported."""
    pixel_cfg = parse_pixel_decoder_cfg(cfg)
    key = resolve_pixel_decoder_type(pixel_cfg.type)
    cls = _PIXEL_DECODER_TYPES[key]
    return cls(pixel_cfg)


__all__ = [
    "FPNPixelDecoder",
    "build_pixel_decoder",
    "list_pixel_decoders",
    "parse_pixel_decoder_cfg",
    "resolve_pixel_decoder_type",
]
