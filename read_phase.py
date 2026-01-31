# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def compare_phase_files(file1, file2):
    """
    加载两个相位文件，并可视化它们的图像及差值。

    参数:
    file1 (str): 第一个 phase.txt 文件的路径。
    file2 (str): 第二个 phase.txt 文件的路径。
    """
    # --- 图形设置，确保中文可以正确显示 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # --- 数据加载 ---
    try:
        print(f"正在加载文件 1: {file1}...")
        # np.genfromtxt 可以自动识别并处理文件中的 'NaN' 字符串
        phase1 = np.genfromtxt(file1)
        print("  - 加载成功。")

        print(f"正在加载文件 2: {file2}...")
        phase2 = np.genfromtxt(file2)
        print("  - 加载成功。")

    except OSError as e:
        print(f"\n错误：加载文件失败。")
        print(f"请确认文件 '{e.filename}' 是否存在于当前目录中。")
        print("您需要先运行主分析脚本两次以生成并重命名两个不同的相位文件。")
        return
    except Exception as e:
        print(f"发生未知错误: {e}")
        return

    # --- 新增: 将读取的数据中 > 1000 的点设置为 NaN ---
    print("\n正在预处理数据: 将 > 1000 的值设置为 NaN...")
    phase1[phase1 > 1000] = np.nan
    phase2[phase2 > 1000] = np.nan
    print("  - 预处理完成。")

    # --- 计算差值 ---
    # Numpy 在进行数组运算时会自动处理 NaN 值
    difference = phase1 - phase2
    print("\n差值计算完成。")

    # --- 可视化 ---
    print("正在生成对比图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('相位图对比分析', fontsize=20)

    # 图 1: 第一个相位文件
    im1 = axes[0].imshow(phase1, cmap='viridis')
    axes[0].set_title(f'相位图 1\n({file1})')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # 图 2: 第二个相位文件
    im2 = axes[1].imshow(phase2, cmap='viridis')
    axes[1].set_title(f'相位图 2\n({file2})')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # 图 3: 差值图
    # 寻找一个对称的颜色范围来更好地显示差异
    diff_abs_max = np.nanmax(np.abs(difference))
    im3 = axes[2].imshow(difference, cmap='coolwarm', vmin=-diff_abs_max, vmax=diff_abs_max)
    axes[2].set_title('差值图 (图1 - 图2)')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 统一关闭坐标轴
    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    # --- 用户需要修改这里的文件名 ---
    # 请根据您重命名的文件名来修改下面的变量
    file_a = r'phase.txt'
    file_b = r'noised/phase.txt'

    print("=" * 50)
    print("相位对比脚本启动")
    print("=" * 50)

    compare_phase_files(file_a, file_b)

    print("\n脚本执行完毕。")
