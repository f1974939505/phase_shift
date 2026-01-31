# 干涉相位展开工具箱（GUI）

用于四步相移干涉的相位处理小工具集，包含简单 GUI：
- 从四幅干涉图 `I1.txt`–`I4.txt` 计算包裹相位并展开（可选降噪）。
- 使用泽尼克多项式拟合并去除背景。
- 相位图对比与差值可视化。

GUI 界面为中文。

## 目录结构
- `phase_unwrap.py`: 基于 `scikit-image` 的相位展开 GUI（四步相移 -> 包裹相位 -> 展开）。
- `phase_unwrap_cv.py`: 基于 OpenCV `phase_unwrapping` 的相位展开 GUI（需要 `opencv-contrib-python`）。
- `zernike_fit.py`: 泽尼克拟合与背景去除 GUI，输出背景与残差并保存系数。
- `read_phase.py`: 两幅相位图对比与差值可视化脚本。
- `replace_nan.py`: 将相位文件中 `> threshold` 的值替换为 `NaN` 的批处理脚本。
- `raw/`: 示例四幅干涉图 `I1.txt`–`I4.txt`。
- `noised/`: 另一组示例数据。

## 环境依赖
- Python 3.8+
- 依赖包：
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `scikit-image`
  - Tkinter（Python 自带的 `tk` GUI 库）
- 可选（仅 `phase_unwrap_cv.py` 需要）：`opencv-contrib-python`

Debian/Ubuntu 安装 Tkinter：
```bash
sudo apt-get update
sudo apt-get install -y python3-tk
# 可选：中文字体（避免图中中文变成方块）
sudo apt-get install -y fonts-noto-cjk
```
安装 Python 包（建议虚拟环境）：
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy scikit-image
# 若使用 OpenCV 版本：
pip install opencv-contrib-python
```

## 显示环境
GUI 使用 `TkAgg` 后端，需要图形界面。无桌面环境可尝试：
```bash
xvfb-run -a python phase_unwrap.py
```

## 数据约定与掩膜规则
- 输入干涉图：`I1.txt`、`I2.txt`、`I3.txt`、`I4.txt`（二维浮点数组）。
- `phase_unwrap.py` 会将数值 > 1.0 视为饱和并加入无效掩膜，随后仅保留**最大连通有效区域**。
- 处理过程中无效点会设为 `NaN` 参与计算屏蔽。
- 输出编码：
  - `phase_unwrap.py` 保存结果时将无效点写为 `100000.0`。
  - `phase_unwrap_cv.py` 直接保存 `NaN`。
- `zernike_fit.py` **期望输入无效点为 `NaN`**。如需处理 `phase_unwrap.py` 的结果，请先用 `replace_nan.py` 转换。
- `read_phase.py` 会把 `> 1000` 的值视作无效并转换为 `NaN`，便于可视化。

## 使用说明
### 1) 相位展开（scikit-image 版本）
运行：
```bash
python phase_unwrap.py
```
步骤：
- 点击“浏览...”选择包含 `I1.txt`–`I4.txt` 的文件夹。
- 可选降噪（中值 / 高斯 / 非局部均值）。
- 点击“运行分析”。程序将计算包裹相位并展开，保存 `phase.txt`。

输出：
- `phase.txt` — 无效点写为 `100000.0`。

### 2) 相位展开（OpenCV 版本）
运行：
```bash
python phase_unwrap_cv.py
```
说明：
- 依赖 `opencv-contrib-python`，要求 `cv2.phase_unwrapping` 可用。
- 输出 `phase.txt` 中无效点保持为 `NaN`。

### 3) 将大值替换为 NaN（可选）
如果你的 `phase.txt` 使用了 `100000.0` 作为无效值，请先转换：
```bash
python replace_nan.py
```
在脚本中修改：
```python
input_folder = r"real"
output_folder = r"real_nan"
```
默认阈值为 `1000.0`，会将大于阈值的元素写成 `NaN`。

### 4) 泽尼克背景去除（zernike_fit.py）
运行：
```bash
python zernike_fit.py
```
步骤：
- 选择 `phase.txt`（无效点需为 `NaN`）。
- 设置“总拟合项数”和“背景定义项数”。
- 可选勾选“去除极端值”并设置百分比（在背景去除后剔除残差极端点）。

输出：
- `background.txt` — 背景相位（保留 `NaN`）。
- `surface.txt` — 残差相位（保留 `NaN`）。
- `zernike_coeffs.txt` — 拟合系数。

备注：结果会裁剪到最小有效矩形区域。

### 5) 相位图对比（read_phase.py）
修改文件路径后运行：
```python
file_a = r'phase.txt'
file_b = r'noised/phase.txt'
```
```bash
python read_phase.py
```
脚本会显示两幅相位图及差值图。

## 排错与建议
- 找不到 Tkinter：安装 `python3-tk`（见上文）。
- 中文显示为方块：安装中文字体（如 `fonts-noto-cjk`）。
- 非局部均值降噪速度较慢，可先用中值或高斯滤波。
- 注意文件名大小写（例如示例中有 `Phase.txt` 与 `phase.txt` 的差异）。

## 许可证
当前仓库未提供许可证文件。如需发布请自行添加。

## 致谢
- 相位展开：`skimage.restoration.unwrap_phase` / OpenCV `phase_unwrapping`。
- 泽尼克拟合：基于 Noll 索引实现，使用 `numpy`/`scipy`。
