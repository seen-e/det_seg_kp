# Det + Seg + Keypoint

统一的实例检测 / 分割 / 关键点（箱体 8 角）训练代码。模型为 MaskFormer 式 query + FPN 像素特征，解码器为 Deformable-DETR 风格的多尺度 deformable attention。

## 1. 任务描述

给定一张场景图（默认输入 **960×768**），预测每个目标实例：


| 输出          | 含义                                    |
| ----------- | ------------------------------------- |
| 类别          | 含 no-object 的分类 logits                |
| 框           | 归一化 `cxcywh`                          |
| 实例 mask     | 与像素特征分辨率一致（默认 stride=4 → **240×192**） |
| 关键点 heatmap | 每实例一张图，融合最多 **8** 个箱角 Gaussian 峰      |


标注角点为 `(x, y, v)`：`v=0` 有效，`v=1` 无效（遮挡/出界）。训练监督用融合 heatmap，不用直接回归 8 个坐标。

**损失概览（Hungarian 匹配后）**

- Det：CE（类别）+ L1 + GIoU（框）；中间 decoder 层有 aux（仅 cls/box）
- Seg：sigmoid focal + Dice（仅最后一层）
- KPS：CenterNet / CornerNet 式 heatmap focal（仅最后一层；bf16 下在 float32 中计算）



## 2. 代码结构

```text
det_seg_kp/
├── modules/
│   ├── det_seg_kp.py          # 主模型：backbone → FPN → deformable decoder → heads
│   ├── config/                # Model / Data / Train 配置
│   ├── backbone/              # ResNet / DINOv2 / CLIP 等 vision tower
│   ├── ops/
│   │   ├── deformable_decoder_layer.py
│   │   └── ms_deform_atten_2D/   # MSDeformAttn（CUDA load + Pytorch grid_sample 回退）
│   ├── loss/
│   │   ├── det_seg_kp_loss.py    # HungarianMatcher + DetSegKPLoss
│   │   ├── sigmoid_focal_loss.py
│   │   ├── dice_loss.py
│   │   └── centernet_heatmap_loss.py
│   └── utils/                 # box、可视化、AMP、分布式、checkpoint
├── datasets/
│   ├── dataset.py             # 读盘、heatmap 生成、DataLoader
│   └── transforms.py          # resize/crop/flip/color
├── scripts/
│   ├── do_train.sh            # 推荐启动入口（单卡 / torchrun）
│   ├── train.py               # 训练主循环
│   └── gen_train_sample.py    # 合成/导出训练样本
├── data/                      # 数据集根目录（需自备或生成）
├── requirements.txt
└── README.md
```

**前向数据流（简图）**

```text
image
  → vision_tower → FPN pixel_decoder → pixel_features (stride 4)
                 → multi-scale memory (≥8) → Deformable decoder → queries
  → class / mask / kp heads（mask、kp = query embed × pixel_features）
```



## 3. 环境配置



### 3.1 依赖

- Python ≥ 3.10（开发环境为 3.12）
- CUDA GPU（训练默认 `bf16` AMP）
- PyTorch ≥ 2.0（需与本机 CUDA 匹配）

```bash
pip install -r requirements.txt
```

`modules/ops/ms_deform_atten_2D` 在首次 import 时通过 `torch.utils.cpp_extension.load` **动态编译** CUDA 算子（产物默认在 `/tmp/torch_extensions`）。编译需要可用的 `nvcc` 与 GPU。失败时会回退到 Pytorch `grid_sample` 实现。

可选：用环境变量指定扩展编译目录：

```bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
```



### 3.2 数据格式

在 `DATA_ROOT`（默认 `./data`）下，每个样本一个 stem（如 `frame_0001`）：

```text
{data_root}/images/{stem}.png
{data_root}/labels/{stem}_instance_mask.png   # 实例 id 图，背景通常为 255
{data_root}/labels/{stem}_train.json          # kps (N,8,3)、labels 等
```

可用 `scripts/gen_train_sample.py` 生成/可视化样本（依赖见该脚本说明）。

默认按 `val_ratio=0.2` 划分 train/val。

## 4. 训练方法



### 4.1 一键启动（推荐）

```bash
bash scripts/do_train.sh
```

- 自动检测 GPU 数：多卡走 `torchrun`，单卡走 `python scripts/train.py`
- 默认开启 wandb（`WANDB=0` 可关）



### 4.2 常用环境变量


| 变量               | 默认          | 说明                               |
| ---------------- | ----------- | -------------------------------- |
| `NPROC_PER_NODE` | 自动检测        | 每节点 GPU 数                        |
| `DATA_ROOT`      | `./data`    | 数据根目录                            |
| `OUTPUT_DIR`     | `./outputs` | checkpoint / vis                 |
| `EPOCHS`         | `50000`     | 训练 epoch 数                       |
| `BATCH_SIZE`     | `4`         | **单卡** batch                     |
| `LR`             | `1e-4`      | 主学习率；backbone 默认 `0.1×`          |
| `WARMUP_RATIO`   | `0`         | cosine 前 warmup 比例；`0` 关闭        |
| `PRECISION`      | `bf16`      | `fp32` / `fp16` / `bf16` / `amp` |
| `WANDB`          | `1`         | 设为 `0` 关闭 wandb                  |


也可把额外参数直接传给 `train.py`：

```bash
bash scripts/do_train.sh --epochs 1000 --batch-size 2 --lr 5e-5
WARMUP_RATIO=0.1 bash scripts/do_train.sh
WANDB=0 DATA_ROOT=./data OUTPUT_DIR=./outputs bash scripts/do_train.sh
```



### 4.3 学习率与优化

- Optimizer：AdamW；`vision_tower` 单独更小学习率（默认 `lr×0.1`）
- Schedule：cosine；`WARMUP_RATIO>0` 时先线性 warmup 再 cosine
- Grad clip：`0.1`
- EMA：默认开启（`decay=0.9999`）



### 4.4 日志与可视化

- 控制台 / wandb：主层分项 + `loss_aux` 汇总 + `loss_total`（不刷每层 aux）
- 周期性写出 GT/Pred 拼图到 `{OUTPUT_DIR}/vis/`（Det / Seg / KPS）



### 4.5 直接调用 [train.py](http://train.py)

```bash
python scripts/train.py --data-root ./data --output-dir ./outputs --precision bf16
torchrun --standalone --nproc_per_node=2 scripts/train.py --batch-size 4
```



## 5. 配置入口

默认超参见 `modules/config/config.py`（`ModelConfig` / `DataConfig` / `TrainConfig`）。命令行与 `do_train.sh` 环境变量会覆盖其中训练相关字段。

主要模型默认：`vision_tower=dict(type="resnet", feature_pyramids=["4x","8x","16x"])`（即 resnet50）、`hidden_dim=1024`、`num_decoder_layers=6`、`num_queries=100`、`pixel_dim=256`。ViT-L 将 `type` 换成具体名字并把 `feature_pyramids` 设为 `["14x"]`。

## 6. 备注

- 输入宽高必须能被 `stride` 整除（默认 960、768、stride=4）。
- Mask / KP 只在 **最后一层** query 上监督；aux 仅 cls/box。
- KP 为正样本定义：heatmap 值 ≥ `peak_thresh`（默认 0.8）的近峰像素，损失为 CenterNet 形式，与 mask 的 RetinaNet `sigmoid_focal_loss` 分开实现。
- 旧 checkpoint 在 decoder / head 结构变更后可能无法直接加载，需重新训练或自行做权重映射。

