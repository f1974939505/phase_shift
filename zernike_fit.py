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

# 设置全局字体以支持中文（如果系统安装了相应字体）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
#  核心分析逻辑 (Zernike Fitting Core)
# =============================================================================

class ZernikeAnalysisCore:
    """
    使用 Noll 索引（j = 1,2,3,...）的标准 Zernike 多项式实现：
      j=1  -> (n,m) = (0, 0)  piston
      j=2  -> (1,-1) tilt X
      j=3  -> (1, 1) tilt Y
      j=4  -> (2,-2) astig 45
      j=5  -> (2, 0) defocus
      j=6  -> (2, 2) astig 0/90

    本软件内部“第 k 项”的顺序略有调整：
      k=1 -> j=1  -> (0, 0)  piston
      k=2 -> j=2  -> (1,-1) tilt X
      k=3 -> j=3  -> (1, 1) tilt Y
      k=4 -> j=5  -> (2, 0) defocus   （与 Noll 顺序相比提前）
      k=5 -> j=4  -> (2,-2) astig 45
      其它 k>=6: j = k

    因此“前 4 项”对应 piston + 两个 tilt + defocus，方便将 defocus 一并视为背景去除。
    """

    @staticmethod
    def noll_to_nm(j: int):
        """将 Noll 索引 j (>=1) 映射到 (n, m)。"""
        if j < 1:
            raise ValueError("Noll 索引 j 必须 >= 1")

        n = 0
        j1 = j - 1
        while j1 > n:
            n += 1
            j1 -= n
        m = -n + 2 * j1
        return int(n), int(m)

    @staticmethod
    def internal_index_to_noll(k: int) -> int:
        """软件内部第 k 项 -> 对应的 Noll 索引 j。

        为了让第 4 项为 defocus，这里交换了 Noll j=4 和 j=5 的顺序：
          k=4 -> j=5 (defocus)
          k=5 -> j=4 (astig 45)
        其它项保持 j=k。
        """
        if k < 1:
            raise ValueError("内部索引 k 必须 >= 1")
        if k == 4:
            return 5
        if k == 5:
            return 4
        return k

    def zernike_radial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """计算径向多项式 R_n^m(rho)，rho 为归一化半径 (0<=rho<=1)。"""
        m = abs(m)
        if (n - m) % 2 != 0:
            return np.zeros_like(rho)
        s_max = (n - m) // 2
        radial_poly = np.zeros_like(rho, dtype=float)
        for s in range(s_max + 1):
            numerator = (-1) ** s * factorial(n - s)
            denominator = (
                factorial(s)
                * factorial((n + m) // 2 - s)
                * factorial((n - m) // 2 - s)
            )
            radial_poly += (numerator / denominator) * (rho ** (n - 2 * s))
        return radial_poly

    def zernike_polynomial(self, j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """返回 Noll 索引为 j 的 Zernike 多项式在极坐标 (rho, theta) 下的值。"""
        if j < 1:
            raise ValueError("Noll 索引 j 必须 >= 1")

        n, m = self.noll_to_nm(j)

        if m == 0:
            return np.sqrt(n + 1) * self.zernike_radial(n, 0, rho)

        radial = self.zernike_radial(n, m, rho)
        if m > 0:
            return np.sqrt(2 * (n + 1)) * radial * np.sin(m * theta)
        else:
            m_abs = abs(m)
            return np.sqrt(2 * (n + 1)) * radial * np.cos(m_abs * theta)

    def fit_and_remove_background(self, phase_file, output_dir, n_terms_fit, selected_terms):
        print("=" * 60)
        print("泽尼克背景去除脚本启动 (内部序号 + 标准 Noll 实现)")
        print("=" * 60)

        # --- 1. 数据加载 ---
        try:
            print(f"正在加载相位文件: {phase_file} ...")
            phase_original = np.genfromtxt(phase_file)
            # 将等于 100000.0 的值视作无效点，并替换为 NaN
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
        center_x = (cols.min() + cols.max()) / 2.0
        center_y = (rows.min() + rows.max()) / 2.0
        radius = max(
            (cols.max() - center_x),
            (rows.max() - center_y),
            (center_x - cols.min()),
            (center_y - rows.min()),
        )
        print(f"  - 检测到有效数据区域: 中心 ({center_x:.1f}, {center_y:.1f}), 半径 {radius:.1f}")

        y, x = np.indices((height, width))
        rho = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / radius
        theta = np.arctan2(y - center_y, x - center_x)

        # 只在单位圆内进行拟合，保证与 Zernike 正交域一致
        aperture_mask = rho <= 1.0
        valid_mask = valid_mask & aperture_mask

        print("  - 坐标归一化完成，并限制在单位圆内。")

        # --- 3. 泽尼克多项式拟合 ---
        print(f"正在构建按内部顺序的前 {n_terms_fit} 项 Zernike 基底并进行拟合...")
        valid_rho = rho[valid_mask]
        valid_theta = theta[valid_mask]
        valid_data = phase_original[valid_mask]

        basis_list = []
        for k in range(1, n_terms_fit + 1):
            j = self.internal_index_to_noll(k)
            zj = self.zernike_polynomial(j, valid_rho, valid_theta)
            basis_list.append(zj)
        basis_matrix = np.vstack(basis_list).T  # 形状: (N_points, n_terms_fit)

        coeffs, _, _, _ = np.linalg.lstsq(basis_matrix, valid_data, rcond=None)
        print(f"  - 拟合完成，获得 {len(coeffs)} 个系数 (按内部序号 1~{n_terms_fit})。")

        # --- 4. 背景重建与误差提取 ---
        print(f"  - 已选择作为背景去除的内部项: {selected_terms if selected_terms else '无'}")

        if not selected_terms:
            print("提示: 未选择任何背景项，将不会去除任何 Zernike 模式，背景默认为 0。")
            phase_background = np.zeros_like(phase_original)
        else:
            background_basis_list = []
            coeff_list = []
            for k in sorted(selected_terms):
                if k < 1 or k > n_terms_fit:
                    print(f"  - 警告: 选择的第 {k} 项超出当前拟合总项数 n_terms_fit={n_terms_fit}，已忽略。")
                    continue
                j = self.internal_index_to_noll(k)
                zj_full = self.zernike_polynomial(j, rho, theta)
                background_basis_list.append(zj_full)
                coeff_list.append(coeffs[k - 1])

            if background_basis_list:
                background_basis = np.stack(background_basis_list, axis=0)  # 形状: (n_bg_selected, H, W)
                background_coeffs = np.array(coeff_list)
                phase_background = np.tensordot(background_coeffs, background_basis, axes=([0], [0]))
                print(f"  - 背景由 {len(background_coeffs)} 项叠加得到。")
            else:
                print("提示: 有选择项但全部被忽略（超出 n_terms_fit），背景设为 0。")
                phase_background = np.zeros_like(phase_original)

        phase_residual = phase_original - phase_background

        print("  - 背景已重建并已从原图中减去。")

        # --- 5. 保存背景和表面误差文件 ---
        print("正在保存背景和表面误差数据...")

        # 处理默认输出目录
        if not output_dir:
            if getattr(sys, 'frozen', False):
                # 如果是打包后的 exe，则使用 exe 所在目录
                output_dir = os.path.dirname(sys.executable)
            else:
                # 如果是普通脚本，则使用脚本所在目录
                output_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"  - 未指定输出目录，将使用默认目录: {output_dir}")

        # 准备背景数据：在无效位置写回 100000.0
        background_to_save = phase_background.copy()
        background_to_save[~valid_mask] = 100000.0

        # 准备表面误差数据：NaN 改为 100000.0
        surface_to_save = phase_residual.copy()
        surface_to_save[np.isnan(surface_to_save)] = 100000.0

        try:
            bg_path = os.path.join(output_dir, "background.txt")
            np.savetxt(bg_path, background_to_save, fmt="%.6f")
            print(f"  - 背景数据已保存至: {bg_path}")

            surf_path = os.path.join(output_dir, "surface.txt")
            np.savetxt(surf_path, surface_to_save, fmt="%.6f")
            print(f"  - 表面误差数据已保存至: {surf_path}")
        except Exception as e:
            print(f"  - 错误: 保存文件失败。 {e}")

        print("脚本执行完毕。")

        return {
            "original": phase_original,
            "background": phase_background,
            "residual": phase_residual,
            "selected_terms": selected_terms,
        }


# =============================================================================
#  GUI 界面
# =============================================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("泽尼克背景去除工具（标准 Noll 索引）")
        self.root.geometry("1100x650")
        self.root.minsize(800, 550)

        self.analysis_core = ZernikeAnalysisCore()

        # 自动根据系统 DPI 调整字体大小
        try:
            scaling_factor = root.tk.call("tk", "scaling")
        except tk.TclError:
            scaling_factor = 1.0

        base_font_size = 10
        self.scaled_font_size = int(base_font_size * float(scaling_factor))
        font_family = "Microsoft YaHei UI"

        style = ttk.Style(root)
        style.configure(".", font=(font_family, self.scaled_font_size))
        style.configure("TButton", font=(font_family, self.scaled_font_size), padding=5)
        style.configure("TLabelframe.Label", font=(font_family, self.scaled_font_size, "bold"))

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 参数设置框 ---
        options_frame = ttk.LabelFrame(main_frame, text="设置", padding="10")
        options_frame.pack(side=tk.TOP, fill=tk.X, expand=False, pady=5)

        # 输入文件
        self.input_file = tk.StringVar()
        ttk.Label(options_frame, text="相位数据文件:").grid(row=0, column=0, padx=5, pady=8, sticky="w")
        ttk.Entry(options_frame, textvariable=self.input_file, font=(font_family, self.scaled_font_size)).grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(options_frame, text="浏览...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        # 输出文件夹
        self.output_dir = tk.StringVar()
        ttk.Label(options_frame, text="输出文件夹 (空则为默认):").grid(
            row=1, column=0, padx=5, pady=8, sticky="w"
        )
        ttk.Entry(options_frame, textvariable=self.output_dir, font=(font_family, self.scaled_font_size)).grid(
            row=1, column=1, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(options_frame, text="浏览...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        # 拟合参数与运行按钮
        param_run_frame = ttk.Frame(options_frame)
        param_run_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)

        self.fit_terms = tk.IntVar(value=15)
        ttk.Label(param_run_frame, text="总拟合项数 (内部序号上限，例如 15)：").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(
            param_run_frame,
            textvariable=self.fit_terms,
            font=(font_family, self.scaled_font_size),
            width=8,
        ).pack(side=tk.LEFT)

        spacer = ttk.Frame(param_run_frame)
        spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.run_button = ttk.Button(param_run_frame, text="运行分析", command=self.start_analysis)
        self.run_button.pack(side=tk.RIGHT)

        options_frame.columnconfigure(1, weight=1)

        # --- Zernike 前 1~8 项说明 + 背景选择 ---
        zernike_info_frame = ttk.LabelFrame(
            main_frame,
            text="Zernike 前 1~8 项（勾选表示作为背景去除）",
            padding="10",
        )
        zernike_info_frame.pack(side=tk.TOP, fill=tk.X, expand=False, pady=5)

        # 定义前 8 项的 (n,m) 与物理含义（按软件内部序号）
        self.term_info = [
            (1, "Piston", 0, 0, "常数项，整体相位平均值"),
            (2, "Tilt X", 1, -1, "沿水平方向的线性倾斜"),
            (3, "Tilt Y", 1, 1, "沿垂直方向的线性倾斜"),
            (4, "Defocus", 2, 0, "焦度/球面项，类似整体聚焦/离焦"),
            (5, "Astig 45°", 2, -2, "一阶散光，主轴约为 45°/135°"),
            (6, "Astig 0/90°", 2, 2, "一阶散光，主轴约为 0°/90°"),
            (7, "Coma 90°", 3, -1, "一阶彗差，主要沿垂直方向"),
            (8, "Coma 0°", 3, 1, "一阶彗差，主要沿水平方向"),
        ]

        self.term_vars = {}
        default_selected = {1, 2, 3, 4}  # 默认去除 Piston + Tilt X + Tilt Y + Defocus

        for k, name, n, m, desc in self.term_info:
            var = tk.BooleanVar(value=(k in default_selected))
            self.term_vars[k] = var
            text = f"第 {k} 项：{name}，(n,m) = ({n}, {m}) —— {desc}"
            ttk.Checkbutton(
                zernike_info_frame,
                text=text,
                variable=var,
            ).pack(anchor="w")

        # --- 日志框架 ---
        log_frame = ttk.LabelFrame(main_frame, text="日志输出", padding="10")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=5)
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            wrap=tk.WORD,
            state="disabled",
            font=(font_family, self.scaled_font_size - 1),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- 结果提示框架 ---
        plot_frame = ttk.LabelFrame(main_frame, text="结果图表", padding="10")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        message_font = (font_family, self.scaled_font_size + 2)
        ttk.Label(
            plot_frame,
            text="分析完成后，图表将在此处弹出一个新窗口。",
            font=message_font,
            anchor="center",
            justify="center",
        ).pack(fill=tk.BOTH, expand=True)

        # 将 stdout 重定向到日志窗口
        sys.stdout = self.RedirectText(self.log_text)

    # ---------------------------
    #  文件选择
    # ---------------------------
    def browse_input(self):
        file_path = filedialog.askopenfilename(
            title="选择 phase.txt 文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if file_path:
            self.input_file.set(file_path)

    def browse_output(self):
        dir_path = filedialog.askdirectory(title="选择输出文件夹")
        if dir_path:
            self.output_dir.set(dir_path)

    # ---------------------------
    #  启动分析
    # ---------------------------
    def start_analysis(self):
        input_file = self.input_file.get()
        if not input_file or not os.path.isfile(input_file):
            print("错误: 请提供一个有效的相位数据文件路径。")
            return

        try:
            n_fit = self.fit_terms.get()
            if n_fit < 1:
                print("错误: 总拟合项数必须 >= 1。")
                return
        except tk.TclError:
            print("错误: 请在总拟合项数输入框中输入有效的整数。")
            return

        # 读取勾选的 Zernike 项
        selected_terms = sorted([k for k, var in self.term_vars.items() if var.get()])
        if selected_terms:
            print(f"本次将作为背景去除的内部 Zernike 项: {selected_terms}")
        else:
            print("提示: 未勾选任何背景项，本次不会去除任何 Zernike 模式。")

        # 禁用按钮 & 清空日志
        self.run_button.config(state="disabled")
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")

        output_dir = self.output_dir.get()

        # 启动后台线程进行计算
        threading.Thread(
            target=self.run_analysis_thread,
            args=(input_file, output_dir, n_fit, selected_terms),
            daemon=True,
        ).start()

    def run_analysis_thread(self, input_file, output_dir, n_fit, selected_terms):
        try:
            results = self.analysis_core.fit_and_remove_background(
                input_file, output_dir, n_terms_fit=n_fit, selected_terms=selected_terms
            )
            if results:
                self.root.after(0, self.show_plot_in_new_window, results)
        except Exception as e:
            print(f"分析过程中发生严重错误: {e}")
        finally:
            self.root.after(0, lambda: self.run_button.config(state="normal"))

    # ---------------------------
    #  绘图
    # ---------------------------
    def show_plot_in_new_window(self, plot_data):
        try:
            print("正在新窗口中生成图表...")

            plot_window = tk.Toplevel(self.root)
            plot_window.title("结果图表")
            plot_window.geometry("1400x700")

            selected_terms = plot_data.get("selected_terms", [])

            fig = Figure(figsize=(18, 6), dpi=100)
            if selected_terms:
                title = f"泽尼克背景去除 (背景: 勾选的 {len(selected_terms)} 项)"
            else:
                title = "泽尼克背景去除 (未去除任何 Zernike 项)"
            fig.suptitle(title, fontsize=self.scaled_font_size + 8)

            axes = fig.subplots(1, 3)
            font_props = {"fontsize": self.scaled_font_size + 4}

            vmax = np.nanmax(plot_data["original"])
            vmin = np.nanmin(plot_data["original"])

            im1 = axes[0].imshow(plot_data["original"], cmap="viridis", vmin=vmin, vmax=vmax)
            axes[0].set_title("原始展开相位", **font_props)
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="相位 (rad)")

            im2 = axes[1].imshow(plot_data["background"], cmap="viridis", vmin=vmin, vmax=vmax)
            if selected_terms:
                axes[1].set_title(f"拟合的背景 (共 {len(selected_terms)} 项)", **font_props)
            else:
                axes[1].set_title("拟合的背景 (未选择任何项，背景=0)", **font_props)
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="相位 (rad)")

            res_vmax = np.nanmax(np.abs(plot_data["residual"]))
            im3 = axes[2].imshow(
                plot_data["residual"],
                cmap="coolwarm",
                vmin=-res_vmax,
                vmax=res_vmax,
            )
            axes[2].set_title("器件表面误差 (残差)", **font_props)
            fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="相位 (rad)")

            for ax in axes.flat:
                ax.axis("off")

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()

            toolbar = NavigationToolbar2Tk(canvas, plot_window, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)

            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            print("图表窗口已生成。")
        except Exception as e:
            print(f"生成图表时发生错误: {e}")

    # ---------------------------
    #  日志输出重定向
    # ---------------------------
    class RedirectText:
        def __init__(self, text_widget):
            self.widget = text_widget

        def write(self, text):
            self.widget.config(state="normal")
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)
            self.widget.config(state="disabled")
            self.widget.update_idletasks()

        def flush(self):
            pass


if __name__ == "__main__":
    # Windows 上设置 DPI 感知，避免界面模糊
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            print("警告: 无法设置 DPI 感知。")

    root = tk.Tk()
    app = App(root)
    root.mainloop()
