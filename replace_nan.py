# -*- coding: utf-8 -*-
"""
批量处理 TXT 数据：
- 从输入文件夹读取 .txt 文件
- 将 > 1000 的值全部替换为 NaN
- 保存到新的输出文件夹中，文件名保持不变
"""

import os
import numpy as np


def replace_large_with_nan(input_dir, output_dir, threshold=1000.0):
    """
    从 input_dir 读取所有 .txt 文件，将其中 > threshold 的数值替换为 NaN，
    并保存到 output_dir 中，文件名保持不变。

    参数
    ----
    input_dir : str
        原始数据所在文件夹路径
    output_dir : str
        处理后数据输出文件夹路径（不存在会自动创建）
    threshold : float
        判定为“无效值”的阈值（默认 1000.0）
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    txt_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(".txt")]

    if not txt_files:
        print("输入文件夹中没有找到任何 .txt 文件。")
        return

    print(f"在 '{input_dir}' 中找到 {len(txt_files)} 个 txt 文件。")
    print(f"处理后将保存到 '{output_dir}' 中。")

    for name in txt_files:
        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, name)

        try:
            data = np.loadtxt(in_path)
        except Exception as e:
            print(f"读取 '{name}' 失败: {e}")
            continue

        # 转为 float，方便写入 NaN
        data = np.asarray(data, dtype=float, copy=True)

        # 找出 > threshold 的元素
        mask = data > threshold
        n_bad = int(mask.sum())

        if n_bad > 0:
            data[mask] = np.nan
            print(f"文件 '{name}': 有 {n_bad} 个值 > {threshold}，已替换为 NaN。")
        else:
            print(f"文件 '{name}': 未发现 > {threshold} 的值。")

        try:
            # 使用 %.6f 输出，NaN 会自动写成字符串 "nan"
            np.savetxt(out_path, data, fmt="%.6f")
            print(f"  -> 已保存到: {out_path}")
        except Exception as e:
            print(f"  -> 保存 '{name}' 失败: {e}")


if __name__ == "__main__":
    # 这里改成你自己的输入/输出文件夹路径
    input_folder = r"real"    # 原来保存 phase.txt / 其它结果的文件夹
    output_folder = r"real_nan"  # 新文件夹（会自动创建）

    # 如果你只想处理某 4 张 I1~I4，可以把上面的筛选条件改成固定列表
    replace_large_with_nan(input_folder, output_folder, threshold=1000.0)
