import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase

# --- 全局设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# --- 核心修改：带垂直偏移的图例处理器 ---
class HandlerMiniBar(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        color = orig_handle.get_facecolor()
        
        # 参数微调
        bar_width = width / 2.2 
        gap = width * 0.1
        
        # 【关键修改】向下偏移量
        # height 通常对应字体高度，向下移动 25% 的高度通常能实现完美居中
        y_offset = height * 0.25 
        
        # 左边的柱子 (较高, 100% 高度)
        rect1 = mpatches.Rectangle(
            xy=(xdescent, ydescent - y_offset), # 这里减去偏移量
            width=bar_width, height=height, 
            facecolor=color, edgecolor='none', transform=trans
        )
        
        # 右边的柱子 (较矮, 60% 高度)
        rect2 = mpatches.Rectangle(
            xy=(xdescent + bar_width + gap, ydescent - y_offset), # 这里减去偏移量
            width=bar_width, height=height * 0.6, 
            facecolor=color, edgecolor='none', transform=trans
        )
        
        return [rect1, rect2]

def draw_rag_comparison_final(metric_name, file_name, y_label, seed=42):
    np.random.seed(seed)
    
    categories = ['VS', 'IS', 'VM', 'IM', 'MM', 'IC']
    x = np.arange(len(categories))
    
    # 柱子宽度
    width = 0.45

    # --- 数据模拟 (SALK > RAG) ---
    salk_data = []
    rag_data = []
    
    for cat in categories:
        base = np.random.uniform(75, 88)
        if cat in ['VS', 'VM']:
            s_val = base + np.random.uniform(2, 5)
            r_val = base - np.random.uniform(5, 12)
        elif cat in ['IS', 'IM', 'MM']:
            s_val = base - np.random.uniform(5, 10)
            r_val = np.random.uniform(5, 15) 
        else: # IC
            s_val = base - np.random.uniform(8, 12)
            r_val = np.random.uniform(25, 40)
        
        salk_data.append(min(100, max(0, s_val)))
        rag_data.append(min(100, max(0, r_val)))

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8.5, 5))

    # 颜色
    color_salk = '#3DA9E6' 
    color_rag = '#F56FA6'  

    rects1 = ax.bar(x - width/2, salk_data, width, label='Qwen-ICL-SALK', 
                    color=color_salk, edgecolor='white', linewidth=0.5)
    rects2 = ax.bar(x + width/2, rag_data, width, label='Qwen-ICL-RAG', 
                    color=color_rag, edgecolor='white', linewidth=0.5)

    # --- 样式细节 ---
    ax.yaxis.grid(True, linestyle='--', color='lightgrey', alpha=0.8, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    
    # 左侧指标
    ax.set_ylabel(y_label, fontweight='bold', fontsize=20)
    # 左侧刻度
    ax.tick_params(axis='both', labelsize=16)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=20, fontweight='bold')
    ax.set_ylim(0, 115) 

    for spine in ax.spines.values():
        spine.set_color('#888888')
        spine.set_linewidth(0.8)
    
    # 柱状图上的数字
    def autolabel(rects, text_color):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', 
                        fontsize=18)
                        # , color=text_color) 
                        # fontweight='bold') # 保持原本注释状态

    autolabel(rects1, color_salk)
    autolabel(rects2, color_rag)

    # --- 自定义图例 (含偏移修正) ---
    proxy_salk = mpatches.Rectangle((0,0), 1, 1, facecolor=color_salk)
    proxy_rag = mpatches.Rectangle((0,0), 1, 1, facecolor=color_rag)
    
    leg = ax.legend(
        [proxy_salk, proxy_rag],
        ['Qwen-ICL-SALK', 'Qwen-ICL-RAG'],
        handler_map={mpatches.Rectangle: HandlerMiniBar()},
        loc='upper left',  # 已修改为 upper left
        ncol=1,            # 已修改为 1 列垂直排列
        frameon=True,
        edgecolor='black',
        fontsize=15,
        handlelength=1.5,
        handleheight=1.2
    )
    
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor('black')

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Generated {file_name}")

# --- 执行生成 ---
draw_rag_comparison_final('SR', 'Figure4-1.png', 'Success Rate (%)', seed=101)
draw_rag_comparison_final('F1', 'Figure4-2.png', 'F1-Score (%)', seed=202)