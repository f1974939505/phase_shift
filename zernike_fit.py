# -*- coding: utf-8 -*-
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import ctypes
from math import factorial

import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 显式指定Tkinter后端
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

# 设置全局字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
#  核心分析逻辑 (Zernike Fitting Core)
# =============================================================================

class ZernikeAnalysisCore:
    def zernike_radial(self, n, m, rho):
        if (n - m) % 2 != 0:
            return np.zeros_like(rho)
        s_max = (n - m) // 2
        radial_poly = np.zeros_like(rho, dtype=float)
        for s in range(s_max + 1):
            numerator = (-1) ** s * factorial(n - s)
            denominator = factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s)
            radial_poly += (numerator / denominator) * (rho ** (n - 2 * s))
        return radial_poly

    def zernike_polynomial(self, j, rho, theta):
        if j < 1:
            raise ValueError("Noll 索引 j 必须 >= 1")
        n = 0
        while (n + 1) * (n + 2) / 2 < j:
            n += 1
        m_abs = j - n * (n + 1) // 2
        if n % 4 < 2:
            m = m_abs if m_abs % 2 != 0 else -m_abs
        else:
            m = m_abs if m_abs % 2 == 0 else -m_abs
        if m == 0:
            return np.sqrt(n + 1) * self.zernike_radial(n, 0, rho)
        radial = self.zernike_radial(n, abs(m), rho)
        if m > 0:
            return np.sqrt(2 * (n + 1)) * radial * np.sin(m * theta)
        else:
            m_abs = abs(m)
            return np.sqrt(2 * (n + 1)) * radial * np.cos(m_abs * theta)

    def fit_and_remove_background(self, phase_file, output_dir, n_terms_fit, n_terms_bg):
        print("=" * 60)
        print("泽尼克背景去除脚本启动 (自包含实现)")
        print("=" * 60)

        # --- 1. 数据加载 ---
        try:
            print(f"正在加载相位文件: {phase_file}...")
            phase_original = np.genfromtxt(phase_file)
            # 将等于100000.0的值视作无效点，并替换为NaN
            phase_original[phase_original == 100000.0] = np.nan
            print("  - 加载成功, 并将 100000.0 转换回 NaN。")
        except OSError:
            print(f"错误: 文件 '{phase_file}' 未找到。")
            return None

        # --- 2. 坐标归一化 ---
        print("正在准备坐标并进行归一化...")
        valid_mask = ~np.isnan(phase_original)
        if not np.any(valid_mask):
            print("错误: 数据文件中没有有效数据点。")
            return None

        height, width = phase_original.shape
        rows, cols = np.where(valid_mask)
        center_x, center_y = (cols.min() + cols.max()) / 2, (rows.min() + rows.max()) / 2
        radius = max((cols.max() - center_x), (rows.max() - center_y),
                     (center_x - cols.min()), (center_y - rows.min()))
        print(f"  - 检测到有效数据区域: 中心 ({center_x:.1f}, {center_y:.1f}), 半径 {radius:.1f}")

        y, x = np.indices((height, width))
        rho = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / radius
        theta = np.arctan2(y - center_y, x - center_x)
        print("  - 坐标归一化完成。")

        # --- 3. 泽尼克多项式拟合 ---
        print(f"正在构建 {n_terms_fit} 项泽尼克基底并进行拟合...")
        valid_rho, valid_theta = rho[valid_mask], theta[valid_mask]
        basis_matrix = np.array([self.zernike_polynomial(j + 1, valid_rho, valid_theta) for j in range(n_terms_fit)]).T
        valid_data = phase_original[valid_mask]
        coeffs, _, _, _ = np.linalg.lstsq(basis_matrix, valid_data, rcond=None)
        print(f"  - 拟合完成，获得 {len(coeffs)} 个系数。")

        # --- 4. 背景重建与误差提取 ---
        print(f"正在使用前 {n_terms_bg} 项系数重建背景...")
        background_basis = np.array([self.zernike_polynomial(j + 1, rho, theta) for j in range(n_terms_bg)])
        background_coeffs = coeffs[:n_terms_bg]
        phase_background = np.tensordot(background_coeffs, background_basis, axes=([0], [0]))
        phase_residual = phase_original - phase_background
        print("  - 背景已重建并已从原图中减去。")

        # --- 5. 保存背景和表面误差文件 ---
        print("正在保存背景和表面误差数据...")

        # 修正：处理默认输出目录
        if not output_dir:
            if getattr(sys, 'frozen', False):
                # 如果是打包后的exe，则使用exe所在目录
                output_dir = os.path.dirname(sys.executable)
            else:
                # 如果是普通脚本，则使用脚本所在目录
                output_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"  - 未指定输出目录，将使用默认目录: {output_dir}")

        # 准备背景数据
        background_to_save = phase_background.copy()
        background_to_save[~valid_mask] = 100000.0  # 应用原始掩码

        # 准备表面误差数据
        surface_to_save = phase_residual.copy()
        surface_to_save[np.isnan(surface_to_save)] = 100000.0  # 将NaN替换为指定值

        try:
            bg_path = os.path.join(output_dir, 'background.txt')
            np.savetxt(bg_path, background_to_save, fmt='%.6f')
            print(f"  - 背景数据已保存至: {bg_path}")

            surf_path = os.path.join(output_dir, 'surface.txt')
            np.savetxt(surf_path, surface_to_save, fmt='%.6f')
            print(f"  - 表面误差数据已保存至: {surf_path}")
        except Exception as e:
            print(f"  - 错误: 保存文件失败。 {e}")

        print("\n脚本执行完毕。")

        return {
            "original": phase_original,
            "background": phase_background,
            "residual": phase_residual,
            "n_terms_bg": n_terms_bg
        }


# =============================================================================
#  GUI 界面
# =============================================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("泽尼克背景去除工具")
        self.root.geometry("1000x600")
        self.root.minsize(700, 500)

        self.analysis_core = ZernikeAnalysisCore()

        try:
            scaling_factor = root.tk.call('tk', 'scaling')
        except tk.TclError:
            scaling_factor = 1.0

        base_font_size = 10
        self.scaled_font_size = int(base_font_size * scaling_factor)
        font_family = "Microsoft YaHei UI"

        style = ttk.Style(root)
        style.configure('.', font=(font_family, self.scaled_font_size))
        style.configure('TButton', font=(font_family, self.scaled_font_size), padding=5)
        style.configure('TLabelframe.Label', font=(font_family, self.scaled_font_size, 'bold'))

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        options_frame = ttk.LabelFrame(main_frame, text="设置", padding="10")
        options_frame.pack(side=tk.TOP, fill=tk.X, expand=False, pady=5)

        # --- 输入文件 ---
        self.input_file = tk.StringVar()
        ttk.Label(options_frame, text="相位数据文件:").grid(row=0, column=0, padx=5, pady=8, sticky="w")
        ttk.Entry(options_frame, textvariable=self.input_file, font=(font_family, self.scaled_font_size)).grid(row=0,
                                                                                                               column=1,
                                                                                                               padx=5,
                                                                                                               pady=5,
                                                                                                               sticky="ew")
        ttk.Button(options_frame, text="浏览...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        # --- 输出文件夹 ---
        self.output_dir = tk.StringVar()
        ttk.Label(options_frame, text="输出文件夹 (空则为默认):").grid(row=1, column=0, padx=5, pady=8, sticky="w")
        ttk.Entry(options_frame, textvariable=self.output_dir, font=(font_family, self.scaled_font_size)).grid(row=1,
                                                                                                               column=1,
                                                                                                               padx=5,
                                                                                                               pady=5,
                                                                                                               sticky="ew")
        ttk.Button(options_frame, text="浏览...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        # --- 拟合参数与运行按钮 ---
        param_run_frame = ttk.Frame(options_frame)
        param_run_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=5)

        self.fit_terms = tk.IntVar(value=15)
        ttk.Label(param_run_frame, text="总拟合项数:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(param_run_frame, textvariable=self.fit_terms, font=(font_family, self.scaled_font_size),
                  width=8).pack(side=tk.LEFT)

        self.bg_terms = tk.IntVar(value=4)
        ttk.Label(param_run_frame, text="背景定义项数:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Entry(param_run_frame, textvariable=self.bg_terms, font=(font_family, self.scaled_font_size), width=8).pack(
            side=tk.LEFT)

        spacer = ttk.Frame(param_run_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.run_button = ttk.Button(param_run_frame, text="运行分析", command=self.start_analysis)
        self.run_button.pack(side=tk.RIGHT)

        options_frame.columnconfigure(1, weight=1)

        # --- 日志框架 ---
        log_frame = ttk.LabelFrame(main_frame, text="日志输出", padding="10")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD, state='disabled',
                                                  font=(font_family, self.scaled_font_size - 1))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- 结果提示框架 ---
        plot_frame = ttk.LabelFrame(main_frame, text="结果图表", padding="10")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        message_font = (font_family, self.scaled_font_size + 2)
        ttk.Label(plot_frame, text="\n\n分析完成后，图表将在此处弹出一个新窗口。\n\n",
                  font=message_font, anchor="center", justify="center").pack(fill=tk.BOTH, expand=True)

        sys.stdout = self.RedirectText(self.log_text)

    def browse_input(self):
        file_path = filedialog.askopenfilename(title="选择 phase.txt 文件",
                                               filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            self.input_file.set(file_path)

    def browse_output(self):
        dir_path = filedialog.askdirectory(title="选择输出文件夹")
        if dir_path:
            self.output_dir.set(dir_path)

    def start_analysis(self):
        input_file = self.input_file.get()
        if not input_file or not os.path.isfile(input_file):
            print("错误: 请提供一个有效的相位数据文件路径。")
            return

        try:
            n_fit = self.fit_terms.get()
            n_bg = self.bg_terms.get()
            if n_fit < n_bg or n_bg < 1:
                print("错误: 拟合项数必须大于等于背景项数，且背景项数至少为1。")
                return
        except tk.TclError:
            print("错误: 请在项数输入框中输入有效的整数。")
            return

        self.run_button.config(state='disabled')
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.config(state='disabled')

        output_dir = self.output_dir.get()

        threading.Thread(target=self.run_analysis_thread, args=(input_file, output_dir, n_fit, n_bg),
                         daemon=True).start()

    def run_analysis_thread(self, input_file, output_dir, n_fit, n_bg):
        try:
            results = self.analysis_core.fit_and_remove_background(input_file, output_dir, n_fit, n_bg)
            if results:
                self.root.after(0, self.show_plot_in_new_window, results)
        except Exception as e:
            print(f"分析过程中发生严重错误: {e}")
        finally:
            self.root.after(0, lambda: self.run_button.config(state='normal'))

    def show_plot_in_new_window(self, plot_data):
        try:
            print("正在新窗口中生成图表...")

            # --- 最终修正：使用Toplevel窗口和面向对象的方式绘图 ---

            # 1. 创建一个新的Toplevel窗口，它是一个独立的弹出窗口
            plot_window = tk.Toplevel(self.root)
            plot_window.title("结果图表")
            plot_window.geometry("1400x700")

            # 2. 创建一个Figure对象 (不使用plt)，这是线程安全的
            fig = Figure(figsize=(18, 6), dpi=100)
            title = f'泽尼克背景去除 (使用前 {plot_data["n_terms_bg"]} 项定义背景)'
            fig.suptitle(title, fontsize=self.scaled_font_size + 8)

            # 3. 在Figure上创建子图
            axes = fig.subplots(1, 3)
            font_props = {'fontsize': self.scaled_font_size + 4}

            vmax = np.nanmax(plot_data["original"])
            vmin = np.nanmin(plot_data["original"])

            im1 = axes[0].imshow(plot_data["original"], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0].set_title('原始展开相位', **font_props)
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='相位 (rad)')

            im2 = axes[1].imshow(plot_data["background"], cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1].set_title(f'拟合的背景 (前 {plot_data["n_terms_bg"]} 项)', **font_props)
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='相位 (rad)')

            res_vmax = np.nanmax(np.abs(plot_data["residual"]))
            im3 = axes[2].imshow(plot_data["residual"], cmap='coolwarm', vmin=-res_vmax, vmax=res_vmax)
            axes[2].set_title('器件表面误差 (残差)', **font_props)
            fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='相位 (rad)')

            for ax in axes.flat:
                ax.axis('off')

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            # 4. 将Figure嵌入到新的Toplevel窗口中
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()

            # 5. 添加导航工具栏
            toolbar = NavigationToolbar2Tk(canvas, plot_window, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)

            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            print("图表窗口已生成。")
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

