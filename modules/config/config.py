"""Configuration for det + seg + keypoint model."""
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    num_classes: int = 1
    num_queries: int = 100
    num_kps: int = 8
    hidden_dim: int = 1024
    num_decoder_layers: int = 6
    num_heads: int = 8
    num_deform_points: int = 4
    pixel_dim: int = 256
    vision_tower: str = "resnet50"
    pretrained_vision_tower: bool = True


@dataclass
class DataConfig:
    """On-disk samples and training-time targets.

    Layout under ``data_root`` (one sample = one stem, e.g. ``frame_0001``)::

        {data_root}/images/{stem}.png
        {data_root}/labels/{stem}_instance_mask.png
        {data_root}/labels/{stem}_train.json

    Model input is ``img_width x img_height`` (default 960 x 768). ``stride``
    sets mask / keypoint-heatmap size to ``(H/stride) x (W/stride)``. Default
    4 matches the FPN pixel-decoder output (ResNet layer1 is already stride 4;
    layer2/3 are fused top-down). Both ``img_width`` and ``img_height`` must
    be divisible by ``stride``.

    ``kp_sigma`` is the Gaussian width in *heatmap* pixels, not input-image
    pixels (e.g. 2.0 at stride=4 is 8 pixels on the input image).
    """

    data_root: str = "./data"
    img_width: int = 960
    img_height: int = 768
    stride: int = 4
    num_workers: int = 8
    num_kps: int = 8
    kp_sigma: float = 2.0
    val_ratio: float = 0.2
    split_seed: int = 42

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
    batch_size: int = 4
    epochs: int = 50000
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_backbone: float = 1e-5  # typically 0.1 * lr
    warmup_epochs_ratio: float = 0.1  # first ratio of epochs; 0 disables (overridden by CLI)
    lr_min_ratio: float = 0.01  # cosine floor = lr * lr_min_ratio
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

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # wandb
    use_wandb: bool = False
    wandb_project: str = "det_seg_kp"
    wandb_run_name: str = ""
    wandb_entity: str = ""

    # loss weights
    loss_cls: float = 2.0
    loss_bbox: float = 5.0
    loss_giou: float = 2.0
    loss_mask: float = 5.0
    loss_dice: float = 5.0
    loss_kp: float = 10.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
