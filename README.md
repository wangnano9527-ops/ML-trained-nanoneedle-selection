# Needle Select：纳米针阵列选择与分割

本项目使用机器学习从荧光显微图像中提取纳米针阵列 mask，并提供可复用的 Python 包、命令行工具和项目模板。当前正式推理模型统一为 **Unified V2 U-Net**。

## 主要功能

- Unified V2 模型推理与滑窗拼接。
- 支持 `single`、`max`、`sum` 三种通道模式。
- 支持 20x、40x、60x 三种输入倍率。
- 未提供倍率时，根据晶格间距或纳米针直径自动估算，并映射到 20x、40x、60x。
- 按预期直径过滤异常连通区域。
- 生成圆形 ROI，并使用晶格拟合校正 ROI 中心。
- 输出 ROI CSV、圆形 mask、overlay、直径统计和半径汇总。
- 提供推理前 `screen`，帮助新使用者检查模型、输入、通道和倍率。

## 快速开始

克隆仓库后，先下载 Git LFS 管理的 V2 模型：

```powershell
git lfs pull
```

安装基础功能：

```powershell
python -m pip install -e .
```

需要训练或模型推理时，安装 ML 依赖：

```powershell
python -m pip install -e ".[ml]"
```

查看项目能力和运行步骤：

```powershell
needle-select describe
needle-select capabilities
needle-select steps
```

## Unified V2 推理

默认模型：

```text
model_registry/unified_v2/needle_unet_unified_v2.pt
```

先编辑 [`configs/example.project.toml`](configs/example.project.toml)，至少确认：

- `[inference].input`：输入图像或图像目录。
- `[inference].channel_mode`：`single`、`max` 或 `sum`。
- `[inference].channel`：使用 `single` 时的通道编号，从 0 开始。
- `[inference].input_magnification`：已知时填写 20、40 或 60；未知时保持未设置。
- `[paths].predictions_dir`：结果输出目录。

正式运行前执行 screen：

```powershell
.\screen.ps1
```

screen 会报告模型和输入路径、图像数量、通道轴、通道数量、实际通道模式、倍率、缩放比例、估算来源和需要操作人员确认的事项。正式 `infer` 也会自动执行同样的检查；screen 未通过时不会开始推理。

开始推理：

```powershell
needle-select run --config configs\example.project.toml --steps infer
```

也可以直接调用脚本：

```powershell
python scripts\infer_needles.py --input D:\your_images --out-dir D:\needle_predictions --channel-mode max --recursive
```

### 通道模式

| 模式 | 用途 |
| --- | --- |
| `single` | 使用一个指定通道，同时设置从 0 开始的 `channel`。 |
| `max` | 对所有通道取最大值；不知道信号通道时建议先使用此模式。 |
| `sum` | 对所有通道求和；适合信号分布在多个通道的情况。 |

普通多页 TIFF 即使没有明确写出 `C` 轴，只要是小型 `QYX/IYX` 页面轴，也会按多通道图像处理。

### 倍率处理

Unified V2 在 40x 模型空间工作：

| 输入倍率 | 模型缩放比例 |
| --- | ---: |
| 20x | 2.0 |
| 40x | 1.0 |
| 60x | 0.6667 |

如果提供倍率，只接受 20x、40x 或 60x。如果未提供，系统先根据晶格间距估算，失败时再使用点直径估算，最后映射到三种受支持倍率之一。所有实际设置和警告都会记录在 `inference_settings.json`。

## 推理输出

每张输入图像可以生成：

- `*_mask_pred.png`：直径过滤后的预测 mask。
- `*_mask_raw.png`：过滤前的原始预测 mask。
- `*_prob.png`：模型概率图。
- `*_overlay.png`：预测结果叠加图。
- `*_circle_rois.csv`：单张图像的圆形 ROI 参数。
- `*_circle_mask.png`：圆形 ROI mask。
- `*_circle_overlay.png`：圆形 ROI 和中心叠加图。

整个批次还会生成：

- `needle_mask_metrics.csv`
- `needle_circle_rois.csv`
- `needle_circle_radius_summary.csv`
- `inference_settings.json`

## 预处理训练数据

原始目录通常包含成对文件：

- `sample_01.tif`
- `sample_01.tif_normalized_mask.png`

运行预处理：

```powershell
python scripts\preprocess_raw_data.py --config configs\preprocess.toml
```

常用调节参数示例：

```powershell
python scripts\preprocess_raw_data.py --lattice-phase-tolerance 0.36 --lattice-min-axial-neighbors 2
```

主要输出：

- `data/images/<sample>_channel1.tif`
- `data/masks/<sample>_mask_clean.png`
- `data/manifest.csv`
- `data/preprocess_summary.json`

mask 清理流程会估算主要纳米针尺寸，删除过小区域，拟合旋转方形晶格，过滤偏离晶格相位的伪点，并保留合理的边缘点和缺失邻居情况。

## 模型训练

生成数据划分、训练 U-Net 并预测：

```powershell
python scripts\make_splits.py --config configs\train.toml
python scripts\train_unet.py --config configs\train.toml
python scripts\predict_masks.py --checkpoint runs\unet_baseline\best.pt --manifest data\manifest.csv --out-dir predictions
```

在新的训练环境中，建议先检查环境：

```powershell
python scripts\check_training_env.py --config configs\train.toml
```

也可以运行完整训练流程：

```powershell
python scripts\run_training_pipeline.py --config configs\train.toml
```

## 在其他项目中调用

安装本项目后，可通过 Python 调用稳定接口：

```python
from pathlib import Path

from needle_select.inference import run_needle_inference

run_needle_inference(
    checkpoint=Path("model_registry/unified_v2/needle_unet_unified_v2.pt"),
    input_path=Path("data/input"),
    output_dir=Path("predictions"),
    channel_mode="max",
    input_magnification=40.0,
    recursive=True,
)
```

建议其他项目依赖本仓库，而不是复制一份 Needle Select 源代码，以便模型和算法只在这里维护。

## 文档

- [`docs/OVERVIEW.md`](docs/OVERVIEW.md)：项目结构与数据流程。
- [`docs/QUICKSTART.md`](docs/QUICKSTART.md)：安装、CLI 和 Python API。
- [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md)：配置字段说明。
- [`docs/INFERENCE_SCREEN.md`](docs/INFERENCE_SCREEN.md)：推理前检查和操作人员输入。
- [`docs/DELIVERY.md`](docs/DELIVERY.md)：迁移与交付规则。
- [`docs/preprocess_parameters.md`](docs/preprocess_parameters.md)：预处理参数说明。
- [`docs/training_setup.md`](docs/training_setup.md)：训练配置说明。

## 测试

运行完整测试：

```powershell
python -m pytest -q tests
```

仅运行圆形 ROI 与晶格中心校正的 7 项专项测试：

```powershell
python -m pytest -q tests\test_inference_rois.py tests\test_lattice.py
```
