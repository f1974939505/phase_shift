# -*- coding: utf-8 -*-
"""相位展开分析工具

- 四步相移获取包裹相位（[-pi, pi]）
- 可选：中值 / 高斯 / OpenCV NL-means 降噪
- 相位展开：使用 OpenCV phase_unwrapping 模块的 HistogramPhaseUnwrapping
  需要 opencv-contrib-python，并且 cv2 中包含 phase_unwrapping 子模块。
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import ctypes  # 处理 DPI 缩放

import numpy as np
import cv2  # OpenCV（要求安装 opencv-contrib-python）

# --- matplotlib 设置（仅用于显示，不参与计算） ---
import matplotlib
matplotlib.use('TkAgg')  # 显式指定 Tkinter 后端
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# SciPy 滤波（中值 / 高斯）
from scipy.ndimage import median_filter, gaussian_filter, label


# =============================================================================
#  OpenCV 辅助函数：NL-means 与相位展开
# =============================================================================


def opencv_nl_means_float(img,
                          h_factor: float = 1.15,
                          patch_size: int = 5,
                          patch_distance: int = 6) -> np.ndarray:
    """对单幅浮点图像做非局部均值滤波（使用 OpenCV fastNlMeansDenoising）。

    步骤：
    1. 将输入 img 映射到 [0,1]，再线性缩放到 [0,255] 的 uint8；
    2. 调用 cv2.fastNlMeansDenoising；
    3. 将结果再映射回原始强度范围。

    参数说明：
    - h_factor: 噪声强度系数，最终传给 fastNlMeansDenoising 的 h ≈ h_factor * sigma * 255；
    - patch_size: 模板窗口尺寸 templateWindowSize（须为奇数）；
    - patch_distance: 搜索半径，对应 searchWindowSize ≈ 2*patch_distance + 1。
    """
    img = np.asarray(img, dtype=np.float32)

    # 估计强度范围
    min_val = float(np.min(img))
    max_val = float(np.max(img))
    if max_val - min_val < 1e-6:
        # 图像几乎为常数，直接返回拷贝
        return img.copy()

    # 归一化到 [0,1]
    img_norm = (img - min_val) / (max_val - min_val)
    img_norm = np.clip(img_norm, 0.0, 1.0)

    # 简单估计噪声标准差
    sigma_est = float(np.std(img_norm))

    # 转换到 0~255 区间
    img_8u = (img_norm * 255.0).astype(np.uint8)

    # OpenCV 的 h 参数是以 0~255 为尺度
    h = h_factor * sigma_est * 255.0
    if h <= 0:
        h = 10.0  # 兜底值

    # OpenCV 要求窗口尺寸为奇数
    templateWindowSize = int(patch_size)
    if templateWindowSize % 2 == 0:
        templateWindowSize += 1

    searchWindowSize = int(2 * patch_distance + 1)
    if searchWindowSize <= templateWindowSize:
        searchWindowSize = templateWindowSize + 2

    denoised_8u = cv2.fastNlMeansDenoising(
        img_8u,
        None,
        h,
        templateWindowSize,
        searchWindowSize,
    )

    # 映回原始浮点范围
    denoised_norm = denoised_8u.astype(np.float32) / 255.0
    denoised = denoised_norm * (max_val - min_val) + min_val
    return denoised


def opencv_unwrap_phase(wrapped_phase: np.ndarray,
                        valid_mask: np.ndarray = None) -> np.ndarray:
    """使用 OpenCV phase_unwrapping 模块进行二维相位展开。

    要求：
    - 已安装 opencv-contrib-python；
    - cv2 中存在子模块 cv2.phase_unwrapping；
    - 输入 wrapped_phase 为 2D 数组，值域在 [-pi, pi]（四步相移 arctan2 的典型输出）。
    - valid_mask 为可选有效区域掩码，True/非零 表示该像素有相位信息，False/0 为无效像素。
      无效像素会通过 shadowMask 从展开过程中屏蔽掉，避免产生伪条纹。
    """
    if not hasattr(cv2, "phase_unwrapping"):
        raise RuntimeError(
            "当前 OpenCV 未包含 phase_unwrapping 模块。\n"
            "请确认安装的是 opencv-contrib-python，而不是 opencv-python。"
        )

    wp = np.asarray(wrapped_phase, dtype=np.float32)
    if wp.ndim != 2:
        raise ValueError(f"wrapped_phase 必须是 2D 数组，当前维度: {wp.ndim}")

    h, w = wp.shape

    # 构造参数对象（优先使用模块形式的 API）
    if hasattr(cv2.phase_unwrapping, "HistogramPhaseUnwrapping_Params"):
        params = cv2.phase_unwrapping.HistogramPhaseUnwrapping_Params()
    elif hasattr(cv2, "phase_unwrapping_HistogramPhaseUnwrapping_Params"):
        # 某些环境下还会额外暴露一个全局别名
        params = cv2.phase_unwrapping_HistogramPhaseUnwrapping_Params()
    else:
        raise RuntimeError(
            "当前 OpenCV 未暴露 HistogramPhaseUnwrapping_Params，无法创建相位展开器。"
        )

    params.width = int(w)
    params.height = int(h)
    # 其他参数（histThresh / nbrOfSmallBins / nbrOfLargeBins）使用默认值即可

    # 创建 HistogramPhaseUnwrapping 展开器
    if hasattr(cv2.phase_unwrapping, "HistogramPhaseUnwrapping") and \
       hasattr(cv2.phase_unwrapping.HistogramPhaseUnwrapping, "create"):
        unwrapper = cv2.phase_unwrapping.HistogramPhaseUnwrapping.create(params)
    elif hasattr(cv2, "phase_unwrapping_HistogramPhaseUnwrapping_create"):
        # 一些旧绑定使用全局工厂函数
        unwrapper = cv2.phase_unwrapping_HistogramPhaseUnwrapping_create(params)
    else:
        raise RuntimeError(
            "当前 OpenCV 未暴露 HistogramPhaseUnwrapping.create / *_create 接口。"
        )

    # 构造 shadowMask：CV_8UC1，非零 = 有效像素，0 = 无效像素
    shadow_mask = None
    if valid_mask is not None:
        vm = np.asarray(valid_mask)
        if vm.shape != wp.shape:
            raise ValueError("valid_mask 形状必须与 wrapped_phase 一致")
        shadow_mask = np.where(vm, 255, 0).astype(np.uint8)

    # Python 签名：
    #   cv.phase_unwrapping.PhaseUnwrapping.unwrapPhaseMap(wrappedPhaseMap[, unwrappedPhaseMap[, shadowMask]]) -> unwrappedPhaseMap
    try:
        if shadow_mask is None:
            unwrapped = unwrapper.unwrapPhaseMap(wp)
        else:
            unwrapped = unwrapper.unwrapPhaseMap(wp, None, shadow_mask)
    except Exception as e:
        raise RuntimeError(f"调用 unwrapPhaseMap 失败: {e}")

    return np.asarray(unwrapped, dtype=np.float64)


# =============================================================================
#  核心分析逻辑
# =============================================================================


class AnalysisCore:
    def load_interferograms_from_txt(self, foldername):
        print(f"步骤 1: 正在从文件夹 '{foldername}' 加载数据并创建掩码...")
        interferograms = []
        saturation_mask = None
        filenames = [os.path.join(foldername, f'I{i}.txt') for i in range(1, 5)]

        for i, filename in enumerate(filenames):
            try:
                data = np.loadtxt(filename)
                data = data.astype(np.float32, copy=False)

                # 饱和掩码：值 > 1 视为无效
                nan_mask = np.isnan(data)
                current_saturation_mask = nan_mask
                # 这些点强度置 0
                data[current_saturation_mask] = 0

                if i == 0:
                    saturation_mask = current_saturation_mask
                else:
                    saturation_mask |= current_saturation_mask

                print(f"  - 成功加载 '{os.path.basename(filename)}' (尺寸: {data.shape})")
                interferograms.append(data)
            except Exception as e:
                print(f"  - 错误: 加载 '{os.path.basename(filename)}' 失败。错误信息: {e}")
                return None, None

        final_mask = saturation_mask
        print(f"  - 最终无效点 (仅基于饱和): {np.sum(final_mask)} 个。")
        return interferograms, final_mask

    @staticmethod
    def keep_largest_valid_region(valid_mask: np.ndarray) -> np.ndarray:
        """保留面积最大的有效连通域（8 连通），返回新的有效掩膜。"""
        if valid_mask is None or not np.any(valid_mask):
            return valid_mask
        structure = np.ones((3, 3), dtype=int)
        labeled, num = label(valid_mask, structure=structure)
        if num <= 1:
            return valid_mask
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest_label = counts.argmax()
        return labeled == largest_label

    def calculate_wrapped_phase(self, interferograms):
        """四步相移法的包裹相位计算，输出范围约为 [-pi, pi]。"""
        i0, i1, i2, i3 = interferograms
        denominator = i0 - i2
        denominator[denominator == 0] = 1e-9
        # [-pi, pi]
        return np.arctan2(i3 - i1, denominator)

    def run_analysis(self, input_dir, output_dir, filter_type):
        total_start_time = time.time()

        # --- 步骤 1: 数据加载 ---
        step1_start_time = time.time()
        interferograms, invalid_mask = self.load_interferograms_from_txt(input_dir)
        if interferograms is None:
            print("\n数据加载失败，程序已终止。")
            return None
        print(f"--- 步骤 1 (数据加载) 完成，耗时: {time.time() - step1_start_time:.4f} 秒 ---\n")

        # 仅保留面积最大的有效连通域，其余标记为无效
        largest_valid = self.keep_largest_valid_region(~invalid_mask)
        if largest_valid is None or not np.any(largest_valid):
            print("错误: 未找到有效的连通区域。")
            return None
        invalid_mask = ~largest_valid
        print(f"  - 已保留面积最大的有效连通域，其他区域设为无效，总无效像素: {np.sum(invalid_mask)}")

        # --- 步骤 1-优化: 降噪 ---
        if filter_type != 'none':
            step_denoise_start = time.time()
            denoised_interferograms = []
            filter_map = {
                'median': '中值滤波',
                'gaussian': '高斯滤波',
                'nl_means': '非局部均值滤波 (OpenCV)',
            }
            print(f"步骤 1-优化: 正在应用降噪...")

            if filter_type == 'median':
                denoised_interferograms = [
                    median_filter(img, size=3) for img in interferograms
                ]
            elif filter_type == 'gaussian':
                denoised_interferograms = [
                    gaussian_filter(img, sigma=1) for img in interferograms
                ]
            elif filter_type == 'nl_means':
                for i, img in enumerate(interferograms):
                    print(
                        f"  - [OpenCV NL-means] 正在处理第 {i + 1}/{len(interferograms)} 幅图像 (这可能需要一些时间)..."
                    )
                    denoised_img = opencv_nl_means_float(
                        img,
                        h_factor=1.15,
                        patch_size=5,
                        patch_distance=6,
                    )
                    denoised_interferograms.append(denoised_img)
            else:
                denoised_interferograms = interferograms

            interferograms = denoised_interferograms
            print(
                f"--- 步骤 1-优化 (降噪) 完成, 耗时: {time.time() - step_denoise_start:.4f} 秒 ---\n"
            )
        else:
            print("步骤 1-优化: 跳过降噪步骤。\n")

        # 最大连通域之外设为 NaN，后续统一作为无效区域
        for idx, img in enumerate(interferograms):
            img = img.astype(float, copy=False)
            img[invalid_mask] = np.nan
            interferograms[idx] = img

        # --- 步骤 2 & 3 & 4: 计算与后处理 ---
        print("步骤 2: 正在计算包裹相位...")
        wrapped_phase = self.calculate_wrapped_phase(interferograms)

        print("步骤 3: 正在使用 OpenCV phase_unwrapping 模块进行相位展开...")
        try:
            # 有效掩码：True 表示像素有效，False 表示饱和/无效
            valid_mask = ~invalid_mask
            wrapped_for_unwrap = np.nan_to_num(wrapped_phase, nan=0.0)
            unwrapped_phase = opencv_unwrap_phase(wrapped_for_unwrap, valid_mask=valid_mask)
        except Exception as e:
            print(f"使用 OpenCV 相位展开时发生错误: {e}")
            return None

        print("步骤 4: 正在进行后处理...")
        wrapped_phase_for_plot = wrapped_phase.astype(float)
        wrapped_phase_for_plot[invalid_mask] = np.nan

        final_phase_for_plot = unwrapped_phase.astype(float)
        final_phase_for_plot[invalid_mask] = np.nan

        # --- 步骤 5: 保存结果 ---
        if not output_dir:
            output_dir_use = (
                os.path.dirname(sys.executable)
                if getattr(sys, 'frozen', False)
                else os.path.dirname(os.path.abspath(__file__))
            )
        else:
            output_dir_use = output_dir

        output_filename = os.path.join(output_dir_use, 'phase.txt')
        print(f"步骤 5: 正在保存最终相位数据至 '{output_filename}'...")

        phase_to_save = final_phase_for_plot.copy()
        # phase_to_save[np.isnan(phase_to_save)] = 100000.0

        try:
            np.savetxt(output_filename, phase_to_save, fmt='%.6f')
            print("  - 保存成功。无效区域已保存为 nan。")
        except Exception as e:
            print(f"  - 保存失败: {e}")

        print(
            f"\n====== 全部流程执行完毕，总耗时: {time.time() - total_start_time:.4f} 秒 ======"
        )

        return {
            "interferogram": interferograms[0],
            "wrapped_phase": wrapped_phase_for_plot,
            "mask": invalid_mask,
            "final_phase": final_phase_for_plot,
        }


# =============================================================================
#  GUI 界面
# =============================================================================


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("相位展开分析工具 (OpenCV phase_unwrapping 版本)")
        self.root.geometry("1200x900")
        self.root.minsize(800, 600)

        self.analysis_core = AnalysisCore()

        # --- DPI 与字体大小自适应 ---
        try:
            scaling_factor = root.tk.call('tk', 'scaling')
        except tk.TclError:
            scaling_factor = 1.0

        base_font_size = 10
        self.scaled_font_size = int(base_font_size * scaling_factor)

        font_family = "Microsoft YaHei UI"

        style = ttk.Style(root)
        style.configure('.', font=(font_family, self.scaled_font_size))
        style.configure('TLabel', font=(font_family, self.scaled_font_size))
        style.configure('TButton', font=(font_family, self.scaled_font_size), padding=5)
        style.configure('TCheckbutton', font=(font_family, self.scaled_font_size))
        style.configure('TLabelframe.Label', font=(font_family, self.scaled_font_size, 'bold'))

        # --- 主框架 ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 选项框架 ---
        options_frame = ttk.LabelFrame(main_frame, text="设置", padding="10")
        options_frame.pack(side=tk.TOP, fill=tk.X, expand=False, pady=5)

        self.input_dir = tk.StringVar()
        ttk.Label(options_frame, text="数据文件夹:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(options_frame, textvariable=self.input_dir,
                  font=(font_family, self.scaled_font_size)).grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(options_frame, text="浏览...", command=self.browse_input).grid(
            row=0, column=2, padx=5, pady=5
        )

        self.output_dir = tk.StringVar()
        ttk.Label(options_frame, text="输出目录 (空则为默认):").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Entry(options_frame, textvariable=self.output_dir,
                  font=(font_family, self.scaled_font_size)).grid(
            row=1, column=1, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(options_frame, text="浏览...", command=self.browse_output).grid(
            row=1, column=2, padx=5, pady=5
        )

        # --- 降噪选项 ---
        self.denoise_var = tk.BooleanVar(value=False)

        self.filter_display_map = {
            '中值滤波': 'median',
            '高斯滤波': 'gaussian',
            '非局部均值滤波': 'nl_means',  # OpenCV NL-means
        }

        self.filter_var = tk.StringVar(value='高斯滤波')

        denoise_check = ttk.Checkbutton(
            options_frame,
            text="启用降噪",
            variable=self.denoise_var,
            command=self.toggle_filter_options,
        )
        denoise_check.grid(row=2, column=0, padx=5, pady=10, sticky="w")

        self.filter_combo = ttk.Combobox(
            options_frame,
            textvariable=self.filter_var,
            values=list(self.filter_display_map.keys()),
            state='disabled',
            font=(font_family, self.scaled_font_size),
        )
        self.root.option_add('*TCombobox*Listbox.font', (font_family, self.scaled_font_size))
        self.filter_combo.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        self.run_button = ttk.Button(options_frame, text="运行分析", command=self.start_analysis)
        self.run_button.grid(row=2, column=2, padx=5, pady=5, sticky="e")

        options_frame.columnconfigure(1, weight=1)

        # --- 日志框架 ---
        log_frame = ttk.LabelFrame(main_frame, text="日志输出", padding="10")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            wrap=tk.WORD,
            state='disabled',
            font=(font_family, self.scaled_font_size - 1),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- 结果提示框架 ---
        plot_frame = ttk.LabelFrame(main_frame, text="结果图表", padding="10")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)

        message_font = (font_family, self.scaled_font_size + 2)
        ttk.Label(
            plot_frame,
            text="\n\n分析完成后，图表将在此处弹出一个新窗口。\n\n",
            font=message_font,
            anchor="center",
            justify="center",
        ).pack(fill=tk.BOTH, expand=True)

        # 重定向 stdout 到日志窗口
        sys.stdout = self.RedirectText(self.log_text)

    def toggle_filter_options(self):
        self.filter_combo.config(state='readonly' if self.denoise_var.get() else 'disabled')

    def browse_input(self):
        dir_path = filedialog.askdirectory(title="选择包含I1-I4.txt的数据文件夹")
        if dir_path:
            self.input_dir.set(dir_path)

    def browse_output(self):
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir.set(dir_path)

    def start_analysis(self):
        input_dir = self.input_dir.get()
        if not input_dir or not os.path.isdir(input_dir):
            print("错误: 请提供一个有效的数据文件夹路径。")
            return

        self.run_button.config(state='disabled')
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='disabled')

        if self.denoise_var.get():
            selected_display_name = self.filter_var.get()
            filter_type = self.filter_display_map.get(selected_display_name, 'gaussian')
        else:
            filter_type = 'none'

        threading.Thread(
            target=self.run_analysis_thread,
            args=(input_dir, self.output_dir.get(), filter_type),
            daemon=True,
        ).start()

    def run_analysis_thread(self, input_dir, output_dir, filter_type):
        try:
            results = self.analysis_core.run_analysis(input_dir, output_dir, filter_type)
            if results:
                self.root.after(0, self.show_plot_in_new_window, results)
        except Exception as e:
            print(f"分析过程中发生严重错误: {e}")
        finally:
            self.root.after(0, lambda: self.run_button.config(state='normal'))

    def show_plot_in_new_window(self, plot_data):
        try:
            print("正在新窗口中生成图表...")

            base_mask = plot_data.get("mask")
            valid_mask = ~base_mask if base_mask is not None else None

            if valid_mask is not None and np.any(valid_mask):
                rows_v, cols_v = np.where(valid_mask)
                rmin, rmax = rows_v.min(), rows_v.max()
                cmin, cmax = cols_v.min(), cols_v.max()
            else:
                rmin = cmin = rmax = cmax = None

            def crop_and_mask(arr, apply_mask=True):
                if rmin is not None:
                    arr = arr[rmin:rmax + 1, cmin:cmax + 1]
                    vm = valid_mask[rmin:rmax + 1, cmin:cmax + 1] if valid_mask is not None else None
                else:
                    vm = valid_mask
                if apply_mask and vm is not None:
                    arr = np.where(vm, arr, np.nan)
                return arr

            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle('数据分析流程', fontsize=self.scaled_font_size + 6)

            font_props = {'fontsize': self.scaled_font_size + 4}

            im0 = axes[0, 0].imshow(crop_and_mask(plot_data["interferogram"]), cmap='gray')
            fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
            axes[0, 0].set_title('(a) 干涉图之一 (已应用滤波)', **font_props)

            im1 = axes[0, 1].imshow(crop_and_mask(plot_data["wrapped_phase"]), cmap='twilight_shifted')
            fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            axes[0, 1].set_title('(b) 包裹相位', **font_props)

            axes[1, 0].imshow(
                plot_data["mask"][rmin:rmax + 1, cmin:cmax + 1] if rmin is not None else plot_data["mask"],
                cmap='gray'
            )
            axes[1, 0].set_title('(c) 无效掩码 (仅饱和)', **font_props)

            im3 = axes[1, 1].imshow(crop_and_mask(plot_data["final_phase"]), cmap='viridis')
            fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
            axes[1, 1].set_title('(d) 最终展开相位', **font_props)

            for ax in axes.flat:
                ax.set_axis_off()

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
            print("图表窗口已关闭。")
        except Exception as e:
            print(f"生成图表时发生错误: {e}")

    class RedirectText:
        def __init__(self, text_widget):
            self.widget = text_widget

        def write(self, text):
            self.widget.config(state='normal')
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)
            self.widget.config(state='disabled')
            self.widget.update_idletasks()

        def flush(self):
            pass


if __name__ == '__main__':
    # Windows DPI 感知设置（防止界面缩放模糊）
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            print("警告: 无法设置DPI感知。")

    root = tk.Tk()
    app = App(root)
    root.mainloop()
