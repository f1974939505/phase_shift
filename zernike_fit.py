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
        """软件内部第 k 项 -> 对应的 Noll 索引 j。"""
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
            # 输入文件无效区域已经是 NaN，不做任何替换
            print("  - 加载成功。")
        except OSError:
            print(f"错误: 文件 '{phase_file}' 未找到。")
            return None

        # --- 2. 坐标归一化 ---
        print("正在准备坐标并进行归一化...")
        # 第一层有效掩膜：非 NaN
        base_valid_mask = ~np.isnan(phase_original)
        if not np.any(base_valid_mask):
            print("错误: 数据文件中没有有效数据点。")
            return None

        height, width = phase_original.shape
        rows0, cols0 = np.where(base_valid_mask)
        center_x = (cols0.min() + cols0.max()) / 2.0
        center_y = (rows0.min() + rows0.max()) / 2.0
        # 使用到最远有效点的距离作为半径，保证所有有效点的 rho<=1
        radius = np.sqrt((cols0 - center_x) ** 2 + (rows0 - center_y) ** 2).max()
        print(f"  - 检测到有效数据区域: 中心 ({center_x:.1f}, {center_y:.1f}), 半径 {radius:.1f}")

        y, x = np.indices((height, width))
        rho = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / radius
        theta = np.arctan2(y - center_y, x - center_x)

        # 不再按圆孔裁切，直接使用输入中的有效掩膜
        valid_mask = base_valid_mask

        print("  - 坐标归一化完成，直接使用有效掩膜（不进行圆形裁剪）。")

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

        # --- 4.1 依据最终 valid_mask 裁剪到最小有效矩形 ---
        rows_v, cols_v = np.where(valid_mask)
        rmin, rmax = rows_v.min(), rows_v.max()
        cmin, cmax = cols_v.min(), cols_v.max()
        print(f"  - 将输出和显示裁剪到行 [{rmin}:{rmax}]，列 [{cmin}:{cmax}] 的有效区域。")

        phase_original = phase_original[rmin:rmax + 1, cmin:cmax + 1]
        phase_background = phase_background[rmin:rmax + 1, cmin:cmax + 1]
        phase_residual = phase_residual[rmin:rmax + 1, cmin:cmax + 1]
        valid_mask = valid_mask[rmin:rmax + 1, cmin:cmax + 1]

        # --- 4.2 计算 PV / RMS 等统计量 ---
        def compute_stats(data: np.ndarray, mask: np.ndarray):
            masked = np.where(mask, data, np.nan)
            if not np.any(mask):
                return None
            vmax = float(np.nanmax(masked))
            vmin = float(np.nanmin(masked))
            pv = vmax - vmin
            mean = float(np.nanmean(masked))
            rms = float(np.sqrt(np.nanmean((masked - mean) ** 2)))
            flat_max = int(np.nanargmax(masked))
            flat_min = int(np.nanargmin(masked))
            max_pos = tuple(np.unravel_index(flat_max, masked.shape))
            min_pos = tuple(np.unravel_index(flat_min, masked.shape))
            return {
                "vmax": vmax,
                "vmin": vmin,
                "pv": pv,
                "mean": mean,
                "rms": rms,
                "max_pos": max_pos,
                "min_pos": min_pos,
            }

        metrics = {
            "original": compute_stats(phase_original, valid_mask),
            "background": compute_stats(phase_background, valid_mask),
            "residual": compute_stats(phase_residual, valid_mask),
        }

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

        # 直接保存裁剪后的数据；无效点保持 NaN
        background_to_save = phase_background.copy()
        surface_to_save = phase_residual.copy()

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
            "metrics": metrics,
            "valid_mask": valid_mask,
            "coeffs": coeffs,
            "n_terms_fit": n_terms_fit,
            "output_dir": output_dir,
        }


# =============================================================================
#  GUI 界面
# =============================================================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("泽尼克背景去除工具（标准 Noll 索引）")
        # 软件默认尺寸：相对于原始 1000x600 放大为宽度*1.5、高度*2
        self.root.geometry("1500x1200")
        self.root.minsize(900, 700)

        self.analysis_core = ZernikeAnalysisCore()

        # 自动根据系统 DPI 调整字体大小
        try:
            scaling_factor = root.tk.call("tk", "scaling")
        except tk.TclError:
            scaling_factor = 1.0

        base_font_size = 10
        self.scaled_font_size = int(base_font_size * float(scaling_factor))
        self.font_family = "Microsoft YaHei UI"

        style = ttk.Style(root)
        style.configure(".", font=(self.font_family, self.scaled_font_size))
        style.configure("TButton", font=(self.font_family, self.scaled_font_size), padding=5)
        style.configure("TLabelframe.Label", font=(self.font_family, self.scaled_font_size, "bold"))
        # 为系数表单独设置 Treeview 的行高和字体，避免内容挤在一起
        style.configure(
            "Coeff.Treeview",
            font=(self.font_family, max(self.scaled_font_size - 1, 8)),
            rowheight=int(self.scaled_font_size * 2.2),
        )
        style.configure(
            "Coeff.Treeview.Heading",
            font=(self.font_family, self.scaled_font_size),
        )

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 参数设置框 ---
        options_frame = ttk.LabelFrame(main_frame, text="设置", padding="10")
        options_frame.pack(side=tk.TOP, fill=tk.X, expand=False, pady=5)

        # 输入文件
        self.input_file = tk.StringVar()
        ttk.Label(options_frame, text="相位数据文件:").grid(row=0, column=0, padx=5, pady=8, sticky="w")
        ttk.Entry(options_frame, textvariable=self.input_file, font=(self.font_family, self.scaled_font_size)).grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )
        ttk.Button(options_frame, text="浏览...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        # 输出文件夹
        self.output_dir = tk.StringVar()
        ttk.Label(options_frame, text="输出文件夹 (空则为默认):").grid(
            row=1, column=0, padx=5, pady=8, sticky="w"
        )
        ttk.Entry(options_frame, textvariable=self.output_dir, font=(self.font_family, self.scaled_font_size)).grid(
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
            font=(self.font_family, self.scaled_font_size),
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

        # 名称映射，供系数表使用
        self.term_name_map = {k: name for (k, name, n, m, desc) in self.term_info}

        # --- 日志框架 ---
        log_frame = ttk.LabelFrame(main_frame, text="日志输出", padding="10")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=5)
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            wrap=tk.WORD,
            state="disabled",
            font=(self.font_family, self.scaled_font_size - 1),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

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
                # 在主线程中更新 UI：显示图和系数表
                self.root.after(0, self.show_plot_in_new_window, results)
                self.root.after(0, self.show_zernike_table_window, results)
        except Exception as e:
            print(f"分析过程中发生严重错误: {e}")
        finally:
            self.root.after(0, lambda: self.run_button.config(state="normal"))

    # ---------------------------
    #  绘图（独立弹窗）
    # ---------------------------
    def show_plot_in_new_window(self, plot_data):
        try:
            print("正在新窗口中生成图表...")

            plot_window = tk.Toplevel(self.root)
            plot_window.title("结果图表")
            plot_window.geometry("1400x700")

            selected_terms = plot_data.get("selected_terms", [])
            metrics = plot_data.get("metrics", {}) or {}
            valid_mask = plot_data.get("valid_mask", None)

            fig = Figure(figsize=(18, 6), dpi=100)
            if selected_terms:
                title = f"泽尼克背景去除 (背景: 勾选的 {len(selected_terms)} 项)"
            else:
                title = "泽尼克背景去除 (未去除任何 Zernike 项)"
            fig.suptitle(title, fontsize=self.scaled_font_size + 8)

            axes = fig.subplots(1, 3)
            font_props = {"fontsize": self.scaled_font_size + 4}

            original = plot_data["original"]
            background = plot_data["background"]
            residual = plot_data["residual"]

            # 使用 valid_mask 对绘图数据做掩膜，圆外无效点设为 NaN
            if valid_mask is not None:
                original_plot = np.where(valid_mask, original, np.nan)
                background_plot = np.where(valid_mask, background, np.nan)
                residual_plot = np.where(valid_mask, residual, np.nan)
            else:
                original_plot = original
                background_plot = background
                residual_plot = residual

            # 颜色范围只按有效区域统计
            vmax = np.nanmax(original_plot)
            vmin = np.nanmin(original_plot)

            # 原始相位
            im1 = axes[0].imshow(original_plot, cmap="viridis", vmin=vmin, vmax=vmax)
            m0 = metrics.get("original")
            if m0 is not None:
                title0 = f"原始展开相位 (RMS = {m0['rms']:.4f} rad)"
            else:
                title0 = "原始展开相位"
            axes[0].set_title(title0, **font_props)
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="相位 (rad)")

            # 拟合背景
            im2 = axes[1].imshow(background_plot, cmap="viridis", vmin=vmin, vmax=vmax)
            m1 = metrics.get("background")
            if m1 is not None:
                if selected_terms:
                    title1 = f"拟合的背景 (共 {len(selected_terms)} 项, RMS = {m1['rms']:.4f} rad)"
                else:
                    title1 = f"拟合的背景 (未选择任何项，背景=0, RMS = {m1['rms']:.4f} rad)"
            else:
                if selected_terms:
                    title1 = f"拟合的背景 (共 {len(selected_terms)} 项)"
                else:
                    title1 = "拟合的背景 (未选择任何项，背景=0)"
            axes[1].set_title(title1, **font_props)
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="相位 (rad)")

            # 残差
            res_vmax = np.nanmax(np.abs(residual_plot))
            # 使用 jet colormap 模拟 MATLAB 默认风格 (红=高, 蓝=低)
            im3 = axes[2].imshow(
                residual_plot,
                cmap="jet",
                vmin=-res_vmax,
                vmax=res_vmax,
            )
            m2 = metrics.get("residual")
            if m2 is not None:
                title2 = f"器件表面误差 (残差, RMS = {m2['rms']:.4f} rad)"
            else:
                title2 = "器件表面误差 (残差)"
            axes[2].set_title(title2, **font_props)
            fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="相位 (rad)")

            # 在三幅图上标记峰/谷值位置，并智能调整文字位置避免出界
            def annotate_peaks(ax, stats, data, label_max="峰", label_min="谷"):
                if stats is None:
                    return
                h, w = data.shape
                (rmax, cmax) = stats["max_pos"]
                (rmin, cmin) = stats["min_pos"]
                vmax_local = stats["vmax"]
                vmin_local = stats["vmin"]

                offset_x = max(w * 0.02, 5)
                offset_y = max(h * 0.02, 5)

                # 峰值标注位置
                if cmax > w * 0.7:
                    x_text_max = cmax - offset_x
                    ha_max = "right"
                else:
                    x_text_max = cmax + offset_x
                    ha_max = "left"
                if rmax > h * 0.8:
                    y_text_max = rmax - offset_y
                    va_max = "top"
                else:
                    y_text_max = rmax + offset_y
                    va_max = "bottom"
                x_text_max = min(max(x_text_max, 0), w - 1)
                y_text_max = min(max(y_text_max, 0), h - 1)

                # 谷值标注位置
                if cmin > w * 0.7:
                    x_text_min = cmin - offset_x
                    ha_min = "right"
                else:
                    x_text_min = cmin + offset_x
                    ha_min = "left"
                if rmin > h * 0.8:
                    y_text_min = rmin - offset_y
                    va_min = "top"
                else:
                    y_text_min = rmin + offset_y
                    va_min = "bottom"
                x_text_min = min(max(x_text_min, 0), w - 1)
                y_text_min = min(max(y_text_min, 0), h - 1)

                # 峰值：圆圈
                ax.scatter([cmax], [rmax], marker="o", s=60, edgecolors="k", facecolors="none")
                ax.text(
                    x_text_max,
                    y_text_max,
                    f"{label_max}:{vmax_local:.3f}",
                    fontsize=self.scaled_font_size,
                    color="black",
                    ha=ha_max,
                    va=va_max,
                )

                # 谷值：叉号
                ax.scatter([cmin], [rmin], marker="x", s=60, color="k")
                ax.text(
                    x_text_min,
                    y_text_min,
                    f"{label_min}:{vmin_local:.3f}",
                    fontsize=self.scaled_font_size,
                    color="black",
                    ha=ha_min,
                    va=va_min,
                )

            annotate_peaks(axes[0], m0, original_plot)
            annotate_peaks(axes[1], m1, background_plot)
            annotate_peaks(axes[2], m2, residual_plot)

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
    #  Zernike 系数表（独立弹窗 + 文件保存）
    # ---------------------------
    def show_zernike_table_window(self, plot_data):
        coeffs = plot_data.get("coeffs")
        if coeffs is None:
            print("警告: 未找到 Zernike 系数，无法显示系数表。")
            return

        n_terms_fit = int(plot_data.get("n_terms_fit", len(coeffs)))
        output_dir = plot_data.get("output_dir", "")

        win = tk.Toplevel(self.root)
        win.title("Zernike 系数表")
        win.geometry("900x500")

        columns = ("k", "j", "n", "m", "name", "coeff")
        tree = ttk.Treeview(win, style="Coeff.Treeview", columns=columns, show="headings")
        tree.heading("k", text="内部序号 k")
        tree.heading("j", text="Noll j")
        tree.heading("n", text="n")
        tree.heading("m", text="m")
        tree.heading("name", text="名称")
        tree.heading("coeff", text="系数 (rad)")

        tree.column("k", width=100, anchor="center", stretch=False)
        tree.column("j", width=80, anchor="center", stretch=False)
        tree.column("n", width=60, anchor="center", stretch=False)
        tree.column("m", width=60, anchor="center", stretch=False)
        tree.column("name", width=220, anchor="w", stretch=True)
        tree.column("coeff", width=260, anchor="e", stretch=True)

        # 滚动条
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        n_rows = min(n_terms_fit, len(coeffs))
        for k in range(1, n_rows + 1):
            j = self.analysis_core.internal_index_to_noll(k)
            n, m = self.analysis_core.noll_to_nm(j)
            name = self.term_name_map.get(k, "")
            coeff_val = float(coeffs[k - 1])
            tree.insert(
                "",
                "end",
                values=(k, j, n, m, name, f"{coeff_val:.6e}"),
            )

        # 同时保存到文件
        self.save_zernike_coeffs_to_file(coeffs, n_rows, output_dir)

    def save_zernike_coeffs_to_file(self, coeffs, n_terms, output_dir):
        if not output_dir:
            if getattr(sys, "frozen", False):
                output_dir = os.path.dirname(sys.executable)
            else:
                output_dir = os.path.dirname(os.path.abspath(__file__))

        path = os.path.join(output_dir, "zernike_coeffs.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# k_internal\tNoll_j\tn\tm\tname\tcoeff(rad)\n")
                for k in range(1, n_terms + 1):
                    j = self.analysis_core.internal_index_to_noll(k)
                    n, m = self.analysis_core.noll_to_nm(j)
                    name = self.term_name_map.get(k, "")
                    coeff_val = float(coeffs[k - 1])
                    f.write(f"{k}\t{j}\t{n}\t{m}\t{name}\t{coeff_val:.10e}\n")
            print(f"  - Zernike 系数已保存至: {path}")
        except Exception as e:
            print(f"  - 错误: 保存 Zernike 系数文件失败。 {e}")

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
