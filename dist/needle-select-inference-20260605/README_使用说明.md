# Needle Select 推理迁移包

这个包只用于把训练好的 U-Net 模型迁移到其它电脑或其它处理流程里，对显微图像自动圈出 nano-needle 区域。

## 包内文件

- `models/needle_unet_best.pt`：训练得到的最佳模型，来自第 62 个 epoch。
- `scripts/infer_needles.py`：推荐使用的推理入口，支持单张图或整个文件夹。
- `scripts/predict_masks.py`：底层滑窗推理函数。
- `needle_select/ml/`：模型结构、图像归一化、配置读取代码。
- `requirements.txt` 和 `requirements-ml.txt`：Python 依赖。

## 安装环境

在新电脑上进入本文件夹，然后创建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

如果新电脑有 NVIDIA GPU，建议按 PyTorch 官网选择对应 CUDA 版本安装 GPU 版 PyTorch，然后再装其它依赖。CPU 也能跑，只是会慢很多。

## 单张图推理

```powershell
.\.venv\Scripts\python.exe scripts\infer_needles.py --checkpoint models\needle_unet_best.pt --input D:\your_images\sample.tif --out-dir D:\needle_predictions --threshold 0.5
```

## 整个文件夹推理

```powershell
.\.venv\Scripts\python.exe scripts\infer_needles.py --checkpoint models\needle_unet_best.pt --input D:\your_images --out-dir D:\needle_predictions --threshold 0.5
```

如果图片在子文件夹里，加 `--recursive`：

```powershell
.\.venv\Scripts\python.exe scripts\infer_needles.py --checkpoint models\needle_unet_best.pt --input D:\your_images --out-dir D:\needle_predictions --threshold 0.5 --recursive
```

## 输入图像要求

- 支持 `.tif`、`.tiff`、`.png`、`.jpg`、`.jpeg`。
- 多页 TIFF 默认使用第 0 页，也就是训练时的 channel1。
- 如果要换通道，用 `--channel`，例如：

```powershell
.\.venv\Scripts\python.exe scripts\infer_needles.py --checkpoint models\needle_unet_best.pt --input D:\your_images --out-dir D:\needle_predictions --channel 1
```

## 输出文件

每张输入图会输出三类文件：

- `*_mask_pred.png`：二值 mask，白色区域是模型圈出的针。其它自动化流程一般接这个文件。
- `*_prob.png`：概率图，越亮代表模型越确信是针。用于调阈值。
- `*_overlay.png`：红色叠加预览图，用于人工快速检查圈得是否合理。

## 阈值怎么选

默认 `--threshold 0.5`。

- 漏针多：降低阈值，例如 `--threshold 0.4` 或 `--threshold 0.35`。
- 背景误圈多：提高阈值，例如 `--threshold 0.6`。
- 想保守一点做后续人工检查：先用 `0.4`，再在后处理里过滤小区域。

改阈值不需要重新训练，只需要重新跑推理。

## 接入其它流程

其它流程最简单的接入方式：

1. 把原图放到一个输入文件夹。
2. 调用 `scripts/infer_needles.py`。
3. 从输出文件夹读取对应的 `*_mask_pred.png`。
4. 用这个 mask 做计数、ROI 提取、轮廓分析、荧光强度统计或后续 ImageJ/Fiji 处理。

输出 mask 与输入图尺寸一致。

## 推荐命令模板

```powershell
.\.venv\Scripts\python.exe scripts\infer_needles.py `
  --checkpoint models\needle_unet_best.pt `
  --input D:\your_images `
  --out-dir D:\needle_predictions `
  --threshold 0.5
```

## 当前模型信息

- 最佳 epoch：62
- 验证集 Dice：0.8124
- Precision：0.8041
- Recall：0.8219

