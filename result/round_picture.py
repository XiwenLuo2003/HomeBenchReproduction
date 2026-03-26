import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR # round_picture.py 放在 HomeBenchReproduction 根目录

RESULT_DIR = os.path.join(PROJECT_ROOT, "round_experiment")
OUTPUT_PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")

# 确保输出目录存在
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- 辅助函数：根据文件名获取图例名称 ---
def get_plot_name(filename):
    name = os.path.basename(filename) # 获取文件名
    if "SALKV7" in name: return "SALK"
    if "SALKV10" in name: return "w/o T-DAR"
    if "SALKV11" in name: return "w/o D-SSD"
    return "baseline"

# --- 核心绘图函数 ---
def draw_round_metrics_plots(home_id):
    print(f"Processing Home ID: {home_id}")

    # 查找所有匹配的 CSV 文件
    search_pattern = os.path.join(RESULT_DIR, f"*_Home_{home_id}.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"Warning: No CSV files found for Home ID {home_id} in {RESULT_DIR}")
        return

    # 存储每个模型的 DataFrame
    model_data = {}
    for f_path in csv_files:
        df = pd.read_csv(f_path)
        # 确保数值保留四位小数 (read_csv 默认会保留，这里只是确保类型正确，避免意外截断)
        for col in ["SR", "P", "Recall", "F1"]:
            if col in df.columns:
                df[col] = df[col].round(4)

        plot_name = get_plot_name(f_path)
        model_data[plot_name] = df
        print(f"Loaded {os.path.basename(f_path)} as '{plot_name}'")

    # 定义要绘制的指标
    metrics = {"SR": "Success Rate", "P": "Precision", "Recall": "Recall", "F1": "F1 Score"}

    # 逐个指标绘制曲线图
    for metric_key, metric_title in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.title(f'Home {home_id} - {metric_title} vs. Round')
        plt.xlabel('Interaction Round')
        plt.ylabel(metric_title)
        plt.grid(True)

        # 遍历所有模型数据并绘制
        for plot_name, df in model_data.items():
            if metric_key in df.columns:
                plt.plot(df['round'], df[metric_key], marker='o', label=plot_name)
            else:
                print(f"Warning: Metric '{metric_key}' not found in data for '{plot_name}'.")

        plt.ylim(0, 1) # 设置 Y 轴范围为 0 到 1
        plt.legend()
        
        # 保存图表
        plot_filename = os.path.join(OUTPUT_PLOT_DIR, f'home_{home_id}_{metric_key.lower()}.png')
        plt.savefig(plot_filename)
        print(f"Saved plot for {metric_title} to: {plot_filename}")
        plt.close() # 关闭当前图形，释放内存

    print(f"Finished plotting for Home ID: {home_id}")

# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot performance metrics over interaction rounds for a given home ID.")
    parser.add_argument("--home_id", type=str, required=True, help="The specific home ID to generate plots for (e.g., '13', '59', '92').")
    args = parser.parse_args()

    # 动态设置 PROJECT_ROOT 为脚本所在目录
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "round_experiment")
    OUTPUT_PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "round_experiment")
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

    draw_round_metrics_plots(args.home_id)
