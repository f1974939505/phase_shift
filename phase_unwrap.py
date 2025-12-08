# -*- coding: utf-8 -*-
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import ctypes  # 导入ctypes库以处理DPI缩放

import numpy as np

# --- 修复matplotlib中文显示问题 ---
import matplotlib

matplotlib.use('TkAgg')  # 显式指定Tkinter后端

# 设置全局字体以支持中文
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 导入 scikit-image 和 Scipy 的功能
from skimage.restoration import unwrap_phase, denoise_nl_means

# estimate_sigma 在新版本 scikit-image 中移动了，做兼容处理
try:
    from skimage.restoration import estimate_sigma
except ImportError:
    # 为旧版 skimage 提供一个简单的回退实现
    def estimate_sigma(image, *args, **kwargs):
        return np.std(image)

from scipy.ndimage import median_filter, gaussian_filter, label


# =============================================================================
#  核心分析逻辑 (从您的脚本修改而来)
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
                current_saturation_mask = data > 1
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
        structure = np.ones((3, 3), dtype=int)  # 8 连通
        labeled, num = label(valid_mask, structure=structure)
        if num <= 1:
            return valid_mask
        counts = np.bincount(labeled.ravel())
        counts[0] = 0  # 忽略背景
        largest_label = counts.argmax()
        return labeled == largest_label

    def calculate_wrapped_phase(self, interferograms):
        i0, i1, i2, i3 = interferograms
        denominator = i0 - i2
        denominator[denominator == 0] = 1e-9
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

        # 保留最大连通有效区域，其余标记为无效
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
            filter_map = {'median': '中值滤波', 'gaussian': '高斯滤波', 'nl_means': '非局部均值滤波'}
            print(f"步骤 1-优化: 正在应用【{filter_map.get(filter_type)}】降噪...")

            if filter_type == 'median':
                denoised_interferograms = [median_filter(img, size=3) for img in interferograms]
            elif filter_type == 'gaussian':
                denoised_interferograms = [gaussian_filter(img, sigma=1) for img in interferograms]
            elif filter_type == 'nl_means':
                for i, img in enumerate(interferograms):
                    print(f"  - 正在处理第 {i + 1}/{len(interferograms)} 幅图像 (这可能需要一些时间)...")
                    sigma_est = np.mean(estimate_sigma(img))
                    denoised_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, patch_size=5,
                                                    patch_distance=6)
                    denoised_interferograms.append(denoised_img)

            interferograms = denoised_interferograms
            print(f"--- 步骤 1-优化 (降噪) 完成, 耗时: {time.time() - step_denoise_start:.4f} 秒 ---\n")
        else:
            print("步骤 1-优化: 跳过降噪步骤。\n")

        # 在最大连通域之外直接设为 NaN，并更新列表
        for idx, img in enumerate(interferograms):
            img = img.astype(float, copy=False)
            img[invalid_mask] = np.nan
            interferograms[idx] = img

        # --- 步骤 2 & 3 & 4: 计算与后处理 ---
        print("步骤 2: 正在计算包裹相位...")
        wrapped_phase = self.calculate_wrapped_phase(interferograms)
        print("步骤 3: 正在进行相位展开...")
        wrapped_for_unwrap = np.nan_to_num(wrapped_phase, nan=0.0)
        unwrapped_phase = unwrap_phase(image=wrapped_for_unwrap)
        print("步骤 4: 正在进行后处理...")
        wrapped_phase_for_plot = wrapped_phase.astype(float)
        wrapped_phase_for_plot[invalid_mask] = np.nan
        final_phase_for_plot = unwrapped_phase.astype(float)
        final_phase_for_plot[invalid_mask] = np.nan

        # --- 步骤 5: 保存结果 ---
        if not output_dir:
            output_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(
                os.path.abspath(__file__))

        output_filename = os.path.join(output_dir, 'phase.txt')
        print(f"步骤 5: 正在保存最终相位数据至 '{output_filename}'...")

        phase_to_save = final_phase_for_plot.copy()
        phase_to_save[np.isnan(phase_to_save)] = 100000.0

        try:
            np.savetxt(output_filename, phase_to_save, fmt='%.6f')
            print("  - 保存成功。无效区域已保存为 100000.000000。")
        except Exception as e:
            print(f"  - 保存失败: {e}")

        print(f"\n====== 全部流程执行完毕，总耗时: {time.time() - total_start_time:.4f} 秒 ======")

        return {
            "interferogram": interferograms[0],
            "wrapped_phase": wrapped_phase_for_plot,
            "mask": invalid_mask,
            "final_phase": final_phase_for_plot
        }


# =============================================================================
#  GUI 界面
# =============================================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("相位展开分析工具")
        self.root.geometry("1200x900")
        self.root.minsize(800, 600)

        self.analysis_core = AnalysisCore()

        # --- DPI与字体大小自适应 ---
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

        # --- 创建主框架 ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 选项框架 ---
        options_frame = ttk.LabelFrame(main_frame, text="设置", padding="10")
        options_frame.pack(side=tk.TOP, fill=tk.X, expand=False, pady=5)

        self.input_dir = tk.StringVar()
        ttk.Label(options_frame, text="数据文件夹:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(options_frame, textvariable=self.input_dir, font=(font_family, self.scaled_font_size)).grid(row=0,
                                                                                                              column=1,
                                                                                                              padx=5,
                                                                                                              pady=5,
                                                                                                              sticky="ew")
        ttk.Button(options_frame, text="浏览...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        self.output_dir = tk.StringVar()
        ttk.Label(options_frame, text="输出目录 (空则为默认):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(options_frame, textvariable=self.output_dir, font=(font_family, self.scaled_font_size)).grid(row=1,
                                                                                                               column=1,
                                                                                                               padx=5,
                                                                                                               pady=5,
                                                                                                               sticky="ew")
        ttk.Button(options_frame, text="浏览...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        # --- 降噪选项 (汉化) ---
        self.denoise_var = tk.BooleanVar(value=False)

        # 1. 创建显示名到内部名的映射
        self.filter_display_map = {
            '中值滤波': 'median',
            '高斯滤波': 'gaussian',
            '非局部均值滤波': 'nl_means'
        }

        # 2. Tkinter变量存储显示名 (中文)
        self.filter_var = tk.StringVar(value='高斯滤波')

        denoise_check = ttk.Checkbutton(options_frame, text="启用降噪", variable=self.denoise_var,
                                        command=self.toggle_filter_options)
        denoise_check.grid(row=2, column=0, padx=5, pady=10, sticky="w")

        # 3. Combobox使用中文列表作为值
        self.filter_combo = ttk.Combobox(options_frame,
                                         textvariable=self.filter_var,
                                         values=list(self.filter_display_map.keys()),
                                         state='disabled',
                                         font=(font_family, self.scaled_font_size))
        self.root.option_add('*TCombobox*Listbox.font', (font_family, self.scaled_font_size))
        self.filter_combo.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        self.run_button = ttk.Button(options_frame, text="运行分析", command=self.start_analysis)
        self.run_button.grid(row=2, column=2, padx=5, pady=5, sticky="e")

        options_frame.columnconfigure(1, weight=1)

        # --- 日志框架 ---
        log_frame = ttk.LabelFrame(main_frame, text="日志输出", padding="10")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD, state='disabled',
                                                  font=(font_family, self.scaled_font_size - 1))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- 结果提示框架 (替换了图表框架) ---
        plot_frame = ttk.LabelFrame(main_frame, text="结果图表", padding="10")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)

        message_font = (font_family, self.scaled_font_size + 2)
        ttk.Label(plot_frame, text="\n\n分析完成后，图表将在此处弹出一个新窗口。\n\n",
                  font=message_font, anchor="center", justify="center").pack(fill=tk.BOTH, expand=True)

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

        # 4. 在启动分析时，将中文显示名转换回内部英文名
        if self.denoise_var.get():
            selected_display_name = self.filter_var.get()
            filter_type = self.filter_display_map.get(selected_display_name, 'gaussian')  # 使用 .get() 更安全
        else:
            filter_type = 'none'

        threading.Thread(target=self.run_analysis_thread, args=(input_dir, self.output_dir.get(), filter_type),
                         daemon=True).start()

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
        # 修正：此函数在主线程中运行，负责创建并显示弹出的图表窗口
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

            # 使用 pyplot 创建一个全新的图表
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle('数据分析流程', fontsize=self.scaled_font_size + 6)

            font_props = {'fontsize': self.scaled_font_size + 4}

            # 绘制并添加Colorbar
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

            # 这个命令会打开一个独立的、可交互的图表窗口
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

