"""Configuration for det + seg + keypoint model."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VisionTowerConfig:
    """Vision backbone and which spatial scales it exposes.

    ``type`` is a registry name (``resnet50``, ``dinov2_vitl14``, …) or a
    family alias (``resnet`` → ``resnet50``). ``feature_pyramids`` selects
    output scales; ``None`` keeps every native scale (ViT-L is typically
    ``["14x"]``).
    """

    type: str = "resnet50"
    feature_pyramids: Optional[List[str]] = None
    pretrained: bool = True


@dataclass
class ObjectQueryDecoderConfig:
    """Object query decoder + FPN memory builder.

    ``type`` selects the implementation (currently only ``deform_detr`` →
    :class:`DeformDETRObjectDecoder`). ``fpn_dim`` / ``scale_keys`` must match
    the pixel decoder FPN. Scales with stride < ``memory_min_stride`` are
    skipped (default 4 keeps every configured FPN level).
    """

    type: str = "deform_detr"
    hidden_dim: int = 1024
    num_layers: int = 6
    num_heads: int = 8
    num_queries: int = 100
    n_points: int = 4
    dropout: float = 0.1
    fpn_dim: int = 256
    scale_keys: List[str] = field(default_factory=lambda: ["4x", "8x", "16x", "32x"])
    memory_min_stride: int = 4


@dataclass
class PixelDecoderConfig:
    """Pixel decoder for mask / keypoint features.

    ``type`` selects the implementation (currently only ``fpn`` →
    :class:`FPNPixelDecoder`). ``in_channels`` maps FPN scale keys to backbone
    channel counts and must match ``vision_tower`` outputs (default = ResNet-50
    4x/8x/16x). ``out_stride`` is the mask/kp feature stride (default 4).
    """

    type: str = "fpn"
    pixel_dim: int = 256
    out_stride: int = 4
    in_channels: Dict[str, int] = field(
        default_factory=lambda: {"4x": 256, "8x": 512, "16x": 1024, "32x": 2048}
    )


@dataclass
class ModelConfig:
    """Top-level model composition: vision tower, pixel decoder, object decoder."""

    num_classes: int = 1
    vision_tower: VisionTowerConfig = field(
        default_factory=lambda: VisionTowerConfig(
            type="resnet",
            feature_pyramids=["4x", "8x", "16x", "32x"],
            pretrained=True,
        )
    )
    pixel_decoder: PixelDecoderConfig = field(default_factory=PixelDecoderConfig)
    object_query_decoder: ObjectQueryDecoderConfig = field(
        default_factory=ObjectQueryDecoderConfig
    )


@dataclass
class DataConfig:
    """On-disk samples and training-time targets.

    Layout under ``data_root`` (from ``gen_and_save_from_tree``)::

        {data_root}/index.json
        {data_root}/{rel_dir}/{stem}.jpg
        {data_root}/{rel_dir}/{stem}_instance_mask.png
        {data_root}/{rel_dir}/{stem}_train.json

    Model input is ``img_width x img_height`` (default 960 x 768). ``stride``
    sets mask / keypoint-heatmap size to ``(H/stride) x (W/stride)``. Default
    4 matches the FPN pixel-decoder output (ResNet layer1 is already stride 4;
    layer2/3 are fused top-down). Both ``img_width`` and ``img_height`` must
    be divisible by ``stride``.

    ``kp_sigma`` is the Gaussian width in *heatmap* pixels, not input-image
    pixels (e.g. 2.0 at stride=4 is 8 pixels on the input image).

    Instance filters (applied after geometric transforms; 0 disables each)::

        min_box_side_hm / min_box_area_hm are in *heatmap* space and scaled by
        ``stride`` / ``stride**2`` to input pixels (larger stride → stricter).
        At stride=4, defaults match short side >= 32 and area >= 32**2.
        ``min_mask_pixels`` counts foreground on the stride-downsampled id-mask.
    """

    data_root: str = "./data"
    img_width: int = 960
    img_height: int = 768
    stride: int = 4
    num_workers: int = 8
    prefetch_factor: int = 2
    num_kps: int = 8
    kp_sigma: float = 2.0
    val_ratio: float = 0.2
    split_seed: int = 42

    # instance size filters (heatmap-space side/area; 0 disables)
    min_box_side_hm: float = 8.0
    min_box_area_hm: float = 64.0
    min_mask_pixels: int = 16

    # geometric augmentation (train only)
    hflip_prob: float = 0.5
    scale_min: float = 0.8
    scale_max: float = 2.0

    # color / photometric augmentation (train only)
    color_brightness: float = 0.2
    color_contrast: float = 0.2
    color_saturation: float = 0.2
    color_hue: float = 0.05


@dataclass
class TrainConfig:
    """Optimizer, schedule, AMP, EMA, logging, and loss weights.

    ``loss_*`` are both criterion weights and Hungarian matcher cost coefficients
    for cls / bbox / giou.
    """

    batch_size: int = 4
    epochs: int = 50000
    # Base LRs; train.py multiplies both by sqrt(total_batch_size / 4).
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_backbone: float = 1e-5  # typically 0.1 * lr
    warmup_epochs_ratio: float = 0.1  # first ratio of epochs; 0 disables (overridden by CLI)
    lr_min_ratio: float = 0.01  # cosine floor = (scaled lr) * lr_min_ratio
    clip_max_norm: float = 0.1
    # fp32 | fp16 (AMP + GradScaler) | bf16 (AMP, no scaler) | amp (bf16 if supported else fp16)
    precision: str = "bf16"
    device: str = "cuda"
    output_dir: str = "./work_dirs"
    log_interval: int = 10
    vis_interval: int = 50
    val_interval: int = 50
    save_interval: int = 5000
    seed: int = 42

    use_ema: bool = True
    ema_decay: float = 0.9999

    use_wandb: bool = False
    wandb_project: str = "det_seg_kp"
    wandb_run_name: str = ""
    wandb_entity: str = ""

    loss_cls: float = 2.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    loss_mask: float = 5.0
    loss_dice: float = 5.0
    loss_kp: float = 10.0


@dataclass
class Config:
    """Root config: model + data + train."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
