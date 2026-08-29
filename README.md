# Det + Seg + Keypoint

统一的实例检测 / 分割 / 关键点（箱体 8 角）训练代码。

架构：MaskFormer 式 object query + FPN 像素特征；解码器为 Deformable-DETR 风格的多尺度 deformable attention。

## 1. 任务

输入场景图（默认 **960×768**），预测每个实例：

| 输出 | 含义 |
| --- | --- |
| 类别 | 含 no-object 的分类 logits |
| 框 | 归一化 `cxcywh` |
| 实例 mask | 与像素特征同分辨率（默认 stride=4 → **240×192**） |
| 关键点 heatmap | 每实例一张图，最多 **8** 个箱角 Gaussian 峰 |

标注角点为 `(x, y, v)`：`v=0` 有效，`v=1` 无效。训练监督用融合 heatmap，不直接回归 8 个坐标。

**损失（Hungarian 匹配后）**

- Det：CE + L1 + GIoU；中间 decoder 层有 aux（仅 cls / box）
- Seg：sigmoid focal + Dice（仅最后一层）
- KPS：RetinaNet 式 sigmoid focal（`pos_thresh` 归一化；仅最后一层）

## 2. 架构与代码结构

```text
image
  → vision_tower          # ResNet / DINOv2 / CLIP …
  → pixel_decoder         # FPN → pixel_features (stride 4) + multi-scale FPN
  → object_query_decoder  # FPN memory (≥8x) + Deformable-DETR decoder
  → heads                 # class / mask_embed×pixel / kp_embed×pixel
```

各子模块均为独立 config + `build_*(cfg)`：

```text
det_seg_kp/
├── modules/
│   ├── det_seg_kp.py              # DetSegKPModel 组装 + DetSegKPHead
│   ├── config/                    # VisionTower / PixelDecoder / ObjectQueryDecoder / Data / Train
│   ├── backbone/                  # build_vision_tower
│   ├── pixel_decoder/             # build_pixel_decoder（FPNPixelDecoder）
│   ├── object_decoder/            # build_object_query_decoder（DeformDETRObjectDecoder）
│   ├── ops/
│   │   ├── deformable_decoder_layer.py
│   │   └── ms_deform_atten_2D/    # MSDeformAttn（CUDA load，失败则 grid_sample）
│   ├── loss/                      # HungarianMatcher + DetSegKPLoss + focal / dice / kp
│   └── utils/                     # box、可视化、AMP、分布式、checkpoint
├── datasets/                      # 读盘、heatmap、aug、DataLoader
├── scripts/
│   ├── do_train.sh                # 推荐入口（单卡 / torchrun）
│   ├── train.py
│   └── gen_train_sample.py
├── data/
├── requirements.txt
└── README.md
```

## 3. 环境

- Python ≥ 3.10（开发环境 3.12）
- CUDA GPU；训练默认 `bf16` AMP
- PyTorch ≥ 2.0（需与本机 CUDA 匹配）

```bash
pip install -r requirements.txt
```

`ms_deform_atten_2D` 首次 import 时通过 `torch.utils.cpp_extension.load` 动态编译（默认目录 `/tmp/torch_extensions`）。需要 `nvcc`；失败时回退到 PyTorch `grid_sample`。

```bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions   # 可选
```

## 4. 数据

`DATA_ROOT`（默认 `./data`）下每个样本一个 stem：

```text
{data_root}/images/{stem}.png
{data_root}/labels/{stem}_instance_mask.png   # 实例 id，背景常为 255
{data_root}/labels/{stem}_train.json          # kps (N,8,3)、labels 等
```

可用 `scripts/gen_train_sample.py` 生成样本。默认 `val_ratio=0.2` 划分 train/val。

输入宽高须能被 `stride` 整除（默认 960、768、stride=4）。

## 5. 训练

### 5.1 一键启动

```bash
bash scripts/do_train.sh
```

自动检测 GPU：多卡 `torchrun`，单卡 `python scripts/train.py`。默认开 wandb（`WANDB=0` 关闭）。

### 5.2 常用环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `NPROC_PER_NODE` | 自动检测 | 每节点 GPU 数 |
| `DATA_ROOT` | `./data` | 数据根目录 |
| `OUTPUT_DIR` | `./work_dirs` | checkpoint / vis |
| `EPOCHS` | `50000` | epoch 数 |
| `BATCH_SIZE` | `4` | **单卡** batch |
| `LR` | `1e-4` | 主学习率；backbone 默认 `0.1×` |
| `WARMUP_RATIO` | `0` | cosine 前 warmup 比例；`0` 关闭 |
| `PRECISION` | `bf16` | `fp32` / `fp16` / `bf16` / `amp` |
| `WANDB` | `1` | 设为 `0` 关闭 |

```bash
bash scripts/do_train.sh --epochs 1000 --batch-size 2 --lr 5e-5
WARMUP_RATIO=0.1 bash scripts/do_train.sh
WANDB=0 DATA_ROOT=./data OUTPUT_DIR=./work_dirs bash scripts/do_train.sh
```

### 5.3 优化与日志

- AdamW；`vision_tower` 单独更小学习率（默认 `lr×0.1`）
- Cosine；`WARMUP_RATIO>0` 时先线性 warmup
- Grad clip `0.1`；EMA 默认开（`decay=0.9999`）
- 控制台 / wandb：主层分项 + `loss_aux` + `loss_total`
- GT/Pred 拼图写到 `{OUTPUT_DIR}/<run>/vis/`

### 5.4 直接调 train.py

```bash
python scripts/train.py --data-root ./data --output-dir ./work_dirs --precision bf16
torchrun --standalone --nproc_per_node=2 scripts/train.py --batch-size 4
```

## 6. 配置

入口：`modules/config/config.py`（`ModelConfig` / `DataConfig` / `TrainConfig`）。CLI 与 `do_train.sh` 会覆盖训练相关字段。

**默认模型拆分**

| Config | 默认要点 |
| --- | --- |
| `vision_tower` | `type="resnet"`（→ resnet50），`feature_pyramids=["4x","8x","16x"]` |
| `pixel_decoder` | `type="fpn"`，`pixel_dim=256`，`out_stride=4`，`in_channels` 对齐 ResNet-50 |
| `object_query_decoder` | `type="deform_detr"`，`hidden_dim=1024`，`num_layers=6`，`num_queries=100`，`scale_keys=["4x","8x","16x"]`，`memory_min_stride=8` |
| `ModelConfig` | `num_classes=1` |

换 ViT-L 时：`vision_tower.type` 换成具体名字，`feature_pyramids=["14x"]`，并同步改 `pixel_decoder.in_channels` 与 `object_query_decoder.scale_keys` / `fpn_dim`。

可用 `modules.backbone.list_vision_towers()` 查看已注册 backbone。

## 7. 备注

- Mask / KP 只在最后一层 query 上监督；aux 仅 cls / box。
- KP 正样本：heatmap ≥ `pos_thresh`（默认 0.01）的像素参与归一化；与 mask 共用 `sigmoid_focal_loss`（mask 为全图 mean）。
- `centernet_heatmap_loss` 为可选实现，当前训练未接入。
- Decoder / head 结构变更后，旧 checkpoint 可能无法直接加载。
