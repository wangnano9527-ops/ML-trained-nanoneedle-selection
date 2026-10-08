# 共享预处理/几何接口 v1（候选）

needle_select.preprocessing_api 提供十四个已核对的纯基础函数。计算仍来自 needle_select.inference，外部只 alias，不复制函数体。

包括：数组轴折叠/通道选择、robust_normalize、倍率/直径算术、float 图像 resize、连通成分直径过滤、最低圆半径、加权中心、等效直径、binary erosion、分位数校验、均值/中位数辅助。

与 Decoding 原函数的完整 AST、NumPy/PIL/SciPy/内建全局绑定及所需同组 helper 均已核对。签名/默认、边界/空值、数值和报错以 golden 测试锁定。API_VERSION=1；preprocessing_source() 报告真实计算文件和 SHA。

不包括 run_needle_inference、load_needle_model、load_model_input_projection、safe_stem、ROI 后处理/导出和覆盖率九个不同实现。接口一致不代表当前 Needle 的 component/自动倍率可替换 Decoding 的旧规则。
