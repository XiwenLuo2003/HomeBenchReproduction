import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
import glob
import numpy as np

# --- 1. 全局字体与参数设置 (极致大字体) ---
config = {
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "mathtext.fontset": 'stix',
    "axes.unicode_minus": False,
    "axes.labelsize": 40,      # 轴标签字号
    "xtick.labelsize": 36,     # X轴刻度字号
    "ytick.labelsize": 36,     # Y轴刻度字号
    "legend.fontsize": 38,     # 图例字号
    "figure.titlesize": 44,
    "axes.linewidth": 3.5      # 坐标轴边框
}
rcParams.update(config)

# 【关键修改】设置 PDF 字体类型为 42 (TrueType)
# 这确保了在 LaTeX/Overleaf 中字体是嵌入的且清晰的，不会变成轮廓
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
RESULT_DIR = os.path.join(PROJECT_ROOT, "result", "round_experiment")
OUTPUT_PLOT_DIR = os.path.join(PROJECT_ROOT, "plots", "round_experiment_final_v7")

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- 自定义图例 Handler 类 ---

class HandlerBarGroup(object):
    def __init__(self, facecolor, edgecolor):
        self.facecolor = facecolor
        self.edgecolor = edgecolor

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        
        pad_y = height * 0.25 
        pad_x = width * 0.25 
        y_shift = height * 0.6 

        draw_h = height - 2 * pad_y
        draw_w_total = width - 2 * pad_x
        
        bar_w = draw_w_total * 0.35 
        gap = draw_w_total * 0.2
        
        base_y = y0 + pad_y - y_shift
        
        rect1 = mpatches.Rectangle([x0 + pad_x, base_y], 
                                   bar_w, draw_h*1.3, 
                                   facecolor=self.facecolor, edgecolor=self.edgecolor, lw=2.0, 
                                   transform=handlebox.get_transform())
        
        rect2 = mpatches.Rectangle([x0 + pad_x + bar_w + gap, base_y], 
                                   bar_w, draw_h * 0.8, 
                                   facecolor=self.facecolor, edgecolor=self.edgecolor, lw=2.0, 
                                   transform=handlebox.get_transform())
                                   
        handlebox.add_artist(rect1)
        handlebox.add_artist(rect2)
        return [rect1, rect2]

class HandlerFallingLine(object):
    def __init__(self, color, marker, is_hollow=False):
        self.color = color
        self.marker = marker
        self.is_hollow = is_hollow

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        
        y_shift = height * 0.6
        cy = y0 + height / 2.0 - y_shift
        half_span = (height * 0.6) / 2.0
        pad_x = width * 0.25
        
        xs = [x0 + pad_x, x0 + width * 0.5, x0 + width - pad_x]
        ys = [cy + half_span, cy, cy - half_span] 
        
        line = mlines.Line2D(xs, ys, color=self.color, linewidth=4.0, 
                             transform=handlebox.get_transform())
        handlebox.add_artist(line)
        
        fc = 'white' if self.is_hollow else self.color
        ec = self.color
        lw = 3.0 if self.is_hollow else 0 
        
        for x, y in zip(xs, ys):
            dot = mlines.Line2D([x], [y], marker=self.marker, 
                                color=self.color, 
                                markerfacecolor=fc, 
                                markeredgecolor=ec,
                                markeredgewidth=lw,
                                markersize=14,  
                                transform=handlebox.get_transform())
            handlebox.add_artist(dot)
            
        return [line]

# --- 辅助函数 ---
def get_plot_name(filename):
    name = os.path.basename(filename)
    if "SALKV7" in name: return "SALK"
    if "SALKV10" in name: return "w/o T-DAR"
    if "SALKV11" in name: return "w/o D-SSD"
    return "baseline"

# --- 核心绘图函数 ---
def draw_round_metrics_plots(home_id, data_type):
    print(f"Processing Home ID: {home_id}, Data Type: {data_type} with PDF generation...")

    # 根据 data_type 构建不同的搜索模式
    if data_type == "single":
        search_pattern = os.path.join(RESULT_DIR, f"*_multi_rounds_of_Home_{home_id}.csv")
    elif data_type == "all":
        search_pattern = os.path.join(RESULT_DIR, f"*_all_rounds_of_Home_{home_id}.csv")
    else:
        raise ValueError("data_type must be 'single' or 'all'")
    
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"Warning: No CSV files found for Home ID {home_id} with data type {data_type}")
        return

    model_data = {}
    MAX_ROUND = 1400
    STEP = 100

    for f_path in csv_files:
        df = pd.read_csv(f_path)
        df = df[df['round'] <= MAX_ROUND]
        df = df[df['round'] % STEP == 0].copy()
        df = df.sort_values('round')
        model_data[get_plot_name(f_path)] = df

    metrics_config = {
        "SR": {"title": "Success Rate (%)", "ylim": (0, 105)},
        "F1": {"title": "F1 Score (%)", "ylim": (0, 105)}
    }

    # --- 2. 颜色与样式配置 ---
    style_map = {
        "w/o D-SSD": {"face": "#aebdf3", "edge": "blue", "label": "w/o D-SSD", "zorder": 2},
        "w/o T-DAR": {"face": "#f3aebc", "edge": "#c10f0f", "label": "w/o T-DAR", "zorder": 3},
        "baseline":  {"color": "#00FF00", "marker": "o", "label": "Qwen-ICL", "zorder": 4, "hollow": False},
        "SALK":      {"color": "#8B0000", "marker": "D", "label": "Qwen-ICL-SALK", "zorder": 5, "hollow": True}
    }

    bar_width = 50 
    bar_offsets = {"w/o D-SSD": -25, "w/o T-DAR": 25}

    all_rounds = sorted(list(set([r for df in model_data.values() for r in df['round']])))
    if not all_rounds: return

    for metric_key, config in metrics_config.items():
        fig, ax = plt.subplots(figsize=(16, 14)) 

        ax.grid(False)
        
        grid_lines_x = [r - 50 for r in all_rounds]
        grid_lines_x.append(all_rounds[-1] + 50)
        
        # ax.vlines(x=grid_lines_x, ymin=0, ymax=105, 
        ax.vlines(x=grid_lines_x, ymin=0, ymax=75, 
                  colors='#D3D3D3', linestyles='-', linewidth=2.0, 
                  zorder=0, clip_on=True)

        # 1. 柱状图
        for name in ["w/o D-SSD", "w/o T-DAR"]:
            if name in model_data:
                df = model_data[name]
                x_coords = df['round'] + bar_offsets[name]
                y_values = df[metric_key] * 100
                
                ax.bar(x_coords, y_values, width=bar_width,
                       color=style_map[name]["face"],
                       edgecolor=style_map[name]["edge"],
                       linewidth=2.0,
                       label=style_map[name]["label"],
                       zorder=style_map[name]["zorder"])

        # 2. 折线图
        for name in ["baseline", "SALK"]:
            if name in model_data:
                df = model_data[name]
                fc = 'white' if style_map[name]["hollow"] else style_map[name]["color"]
                ec = style_map[name]["color"]
                lw_marker = 3.0 if style_map[name]["hollow"] else 0
                
                ax.plot(df['round'], df[metric_key] * 100,
                        color=style_map[name]["color"],
                        marker=style_map[name]["marker"],
                        markersize=22,
                        linewidth=3.0,
                        markerfacecolor=fc, 
                        markeredgecolor=ec,
                        markeredgewidth=lw_marker,
                        label=style_map[name]["label"],
                        zorder=style_map[name]["zorder"])

        # --- 轴设置 ---
        ax.set_xticks(all_rounds)
        ax.set_xticklabels(all_rounds, rotation=45, ha='right') 
        
        # ax.set_ylim(0, 105)
        ax.set_ylim(0, 75)
        ax.tick_params(axis='y', direction='in', length=12, width=3.0, pad=10)
        ax.tick_params(axis='x', direction='out', length=12, width=3.0, pad=10)
        
        ax.set_ylabel(config["title"], fontweight='bold', labelpad=2)
        ax.set_xlim(left=all_rounds[0]-50, right=all_rounds[-1]+50)

        for spine in ax.spines.values():
            spine.set_linewidth(3.0)
            spine.set_edgecolor('black')

        # --- 3. 图例 ---
        custom_handles = [
            mpatches.Rectangle((0,0),1,1),
            mpatches.Rectangle((0,0),1,1),
            mlines.Line2D([], []),
            mlines.Line2D([], [])
        ]
        
        custom_labels = ["w/o D-SSD", "w/o T-DAR", "Qwen-ICL", "Qwen-ICL-SALK"]
        
        handler_map = {
            custom_handles[0]: HandlerBarGroup(style_map["w/o D-SSD"]["face"], style_map["w/o D-SSD"]["edge"]),
            custom_handles[1]: HandlerBarGroup(style_map["w/o T-DAR"]["face"], style_map["w/o T-DAR"]["edge"]),
            custom_handles[2]: HandlerFallingLine(style_map["baseline"]["color"], style_map["baseline"]["marker"], False),
            custom_handles[3]: HandlerFallingLine(style_map["SALK"]["color"], style_map["SALK"]["marker"], True)
        }

        legend = ax.legend(custom_handles, custom_labels,
                           handler_map=handler_map,
                           loc='upper center', 
                           bbox_to_anchor=(-0.05, -0.22, 1.1, 0.1),
                           mode="expand",
                           ncol=4,
                           frameon=True, edgecolor='black', 
                           framealpha=1, fancybox=False,
                           handleheight=2.0, 
                           handlelength=2.0,
                           handletextpad=-0.4,
                           borderpad=0.6,
                           columnspacing=1.0)

        legend.get_frame().set_linewidth(3.0)

        # 左侧空白
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.22) 

        # --- 【关键修改】保存 PNG 和 PDF ---
        # 1. 保存 PDF (推荐 Overleaf 使用，体积小，无限放大不失真)
        pdf_filename = os.path.join(OUTPUT_PLOT_DIR, f'home_{home_id}_{data_type}_{metric_key.lower()}.pdf')
        print(f"Saving PDF: {pdf_filename}")
        plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')

        # 2. 保存 PNG (用于快速预览)
        png_filename = os.path.join(OUTPUT_PLOT_DIR, f'home_{home_id}_{data_type}_{metric_key.lower()}.png')
        print(f"Saving PNG: {png_filename}")
        plt.savefig(png_filename, dpi=300)
        
        plt.close()

    print(f"Finished Home {home_id} for data type {data_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_id", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True, choices=["single", "all"], help="Type of data to plot: 'single' for single-instruction data, 'all' for all-instruction data.")
    args = parser.parse_args()
    draw_round_metrics_plots(args.home_id, args.data_type)