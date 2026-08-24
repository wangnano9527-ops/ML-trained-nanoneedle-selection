# Inference Screen And Operator Inputs

新使用者不需要先读代码。配置好项目后，先运行：

```powershell
.\screen.ps1
```

或：

```powershell
needle-select screen --config configs\example.project.toml
```

## 什么时候运行

- 安装或换电脑后先运行 `doctor`，检查 Python、PyTorch 和环境。
- 第一次推理前运行 `screen`。
- 更换显微镜、输入目录、通道布局、倍率或模型后，再运行一次 `screen`。
- 通过 `needle-select run ... --steps infer` 正式运行时，项目会自动执行同样的 screen；未通过会阻止推理。

不需要在训练或预处理的每一步都运行 screen。它是 inference 输入门禁；`doctor` 才是整个项目的环境检查。

## 运行人员必须确认的输入

| 输入 | 可选值/格式 | 不知道时怎么办 |
| --- | --- | --- |
| 模型 | unified v2 checkpoint | 使用仓库默认 V2，并确认已执行 `git lfs pull` |
| 图像 | TIFF/PNG/JPEG 文件或目录 | screen 会报告文件数量与样例轴信息 |
| 通道模式 | `single` / `max` / `sum` | 默认 `max`；信号分散到各通道时用 `sum` |
| 单通道编号 | 从 0 开始，例如 `0`、`1` | 仅 `single` 需要；screen 会报告检测到的通道数 |
| 倍率 | 20x / 40x / 60x | 留空，系统从晶格间距或直径估算并映射到三者之一 |
| 输出目录 | 可写入的目录 | 默认由项目配置指定 |

`screen` 会抽样读取图像但不会加载模型做正式推理，也不会产生预测结果。请检查每个样例显示的 `axes`、`channels`、选定倍率、scale 和估算来源。如果倍率估算失败，系统回退到 40x 并给出警告；生产运行前应人工确认。

## 正式输出

- `*_mask_pred.png`、`*_prob.png`、`*_overlay.png`
- `*_circle_rois.csv`、`needle_circle_rois.csv`
- `*_circle_mask.png`、`*_circle_overlay.png`
- `needle_mask_metrics.csv`
- `needle_circle_radius_summary.csv`
- `inference_settings.json`，记录每张图实际使用的通道、倍率、scale、直径和警告

## 专项回归测试

圆形 ROI 与晶格中心校正由 7 项专项测试覆盖：

```powershell
python -m pytest -q tests\test_inference_rois.py tests\test_lattice.py
```

通道模式、普通多页 TIFF、倍率映射和 screen 另有独立测试。
