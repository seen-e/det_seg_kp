"""Object query decoder factory."""
from __future__ import annotations

from typing import Any, Mapping, Union

from modules.config import ObjectQueryDecoderConfig
from modules.object_decoder.object_query_decoder import ObjectQueryDecoder

ObjectQueryDecoderSpec = Union[ObjectQueryDecoderConfig, Mapping[str, Any]]


def parse_object_query_decoder_cfg(cfg: ObjectQueryDecoderSpec) -> ObjectQueryDecoderConfig:
    """Normalize dict / dataclass into :class:`ObjectQueryDecoderConfig`."""
    if isinstance(cfg, ObjectQueryDecoderConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return ObjectQueryDecoderConfig(**dict(cfg))
    raise TypeError(
        f"object_query_decoder must be ObjectQueryDecoderConfig or dict; "
        f"got {type(cfg).__name__}"
    )


def build_object_query_decoder(cfg: ObjectQueryDecoderSpec) -> ObjectQueryDecoder:
    """Build an :class:`ObjectQueryDecoder` from config (only argument)."""
    return ObjectQueryDecoder(parse_object_query_decoder_cfg(cfg))


__all__ = [
    "ObjectQueryDecoder",
    "build_object_query_decoder",
    "parse_object_query_decoder_cfg",
]
