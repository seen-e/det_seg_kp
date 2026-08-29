"""Object query decoder factory."""
from __future__ import annotations

from typing import Any, Mapping, Union

import torch.nn as nn

from ..config import ObjectQueryDecoderConfig
from .object_query_decoder import DeformDETRObjectDecoder

ObjectQueryDecoderSpec = Union[ObjectQueryDecoderConfig, Mapping[str, Any], str]

# Registry name → class. Only DeformDETR is wired for now.
_OBJECT_DECODER_TYPES = {
    "deform_detr": DeformDETRObjectDecoder,
}

_FAMILY_DEFAULTS = {
    "deformable_detr": "deform_detr",
}


def list_object_query_decoders() -> list[str]:
    return sorted(_OBJECT_DECODER_TYPES)


def parse_object_query_decoder_cfg(cfg: ObjectQueryDecoderSpec) -> ObjectQueryDecoderConfig:
    """Normalize dict / str / dataclass into :class:`ObjectQueryDecoderConfig`."""
    if isinstance(cfg, ObjectQueryDecoderConfig):
        return cfg
    if isinstance(cfg, str):
        return ObjectQueryDecoderConfig(type=cfg)
    if isinstance(cfg, Mapping):
        if "type" not in cfg:
            raise ValueError("object_query_decoder dict needs 'type'")
        return ObjectQueryDecoderConfig(**dict(cfg))
    raise TypeError(
        f"object_query_decoder must be ObjectQueryDecoderConfig, dict, or str; "
        f"got {type(cfg).__name__}"
    )


def resolve_object_query_decoder_type(type_name: str) -> str:
    """Map ``type`` to a registry key (``deformable_detr`` → ``deform_detr``)."""
    key = type_name.lower().strip()
    if key in _OBJECT_DECODER_TYPES:
        return key
    if key in _FAMILY_DEFAULTS:
        return _FAMILY_DEFAULTS[key]
    raise ValueError(
        f"Unknown object_query_decoder type '{type_name}'. "
        f"Supported: {', '.join(list_object_query_decoders())}. "
        f"Aliases: {', '.join(sorted(_FAMILY_DEFAULTS))}"
    )


def build_object_query_decoder(cfg: ObjectQueryDecoderSpec) -> nn.Module:
    """Build from config. Currently only ``type="deform_detr"`` is supported."""
    decoder_cfg = parse_object_query_decoder_cfg(cfg)
    key = resolve_object_query_decoder_type(decoder_cfg.type)
    cls = _OBJECT_DECODER_TYPES[key]
    return cls(decoder_cfg)


__all__ = [
    "DeformDETRObjectDecoder",
    "build_object_query_decoder",
    "list_object_query_decoders",
    "parse_object_query_decoder_cfg",
    "resolve_object_query_decoder_type",
]
