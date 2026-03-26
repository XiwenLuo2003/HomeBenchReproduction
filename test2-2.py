import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase

# --- 全局设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16 # 从 12 调大到 16

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

def draw_rag_comparison_final(metric_name, file_name, y_label):
    
    categories = ['VS', 'IS', 'VM', 'IM', 'MM', 'IC']
    x = np.arange(len(categories))
    
    # 柱子宽度
    width = 0.45

    # --- 真实数据录入 ---
    if metric_name == 'SR':
        # SR 数据
        # Qwen-ICL-SALK
        salk_data = [74.96, 82.48, 40.0, 0.0, 0.59, 75.1]
        # Qwen-ICL-RAG
        rag_data  = [20.86, 77.31, 7.35, 0.00, 0.44, 50.48]
    
    elif metric_name == 'F1':
        # F1 数据
        # Qwen-ICL-SALK
        salk_data = [74.94, 82.43, 72.58, 12.44, 34.43, 75.06]
        # Qwen-ICL-RAG
        rag_data  = [20.83, 77.17, 31.01, 29.71, 32.49, 42.34]
    
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")

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
    ax.set_ylabel(y_label, fontweight='bold', fontsize=20) # 从 20 调大到 22
    # 左侧刻度
    ax.tick_params(axis='both', labelsize=18) # 从 16 调大到 18
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=22, fontweight='bold') # 从 20 调大到 22
    
    # 动态调整Y轴上限，给数值标签留出空间
    max_val = max(max(salk_data), max(rag_data))
    ax.set_ylim(0, max_val * 1.15) 

    for spine in ax.spines.values():
        spine.set_color('#888888')
        spine.set_linewidth(0.8)
    
    # 柱状图上的数字
    def autolabel(rects, text_color):
        for rect in rects:
            height = rect.get_height()
            # 对于非常小的值（比如0），可以根据需求选择是否显示或特殊处理
            # 这里保持显示两位小数
            label_text = f'{height:.2f}'
            
            ax.annotate(label_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', 
                        fontsize=18) # 从 14 调大到 16

    autolabel(rects1, color_salk)
    autolabel(rects2, color_rag)

    # --- 自定义图例 (含偏移修正) ---
    proxy_salk = mpatches.Rectangle((0,0), 1, 1, facecolor=color_salk)
    proxy_rag = mpatches.Rectangle((0,0), 1, 1, facecolor=color_rag)
    
    leg = ax.legend(
        [proxy_salk, proxy_rag],
        ['Qwen-ICL-SALK', 'Qwen-ICL-RAG'],
        handler_map={mpatches.Rectangle: HandlerMiniBar()},
        loc='upper left',  
        bbox_to_anchor=(0.34, 1.1),
        ncol=1,           
        frameon=True,
        edgecolor='black',
        fontsize=18, # 从 15 调大到 18
        handlelength=1.5,
        handleheight=1.2
    )
    
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_edgecolor('black')

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    # 【新增】生成 PDF 文件
    pdf_file_name = file_name.replace('.png', '.pdf')
    plt.savefig(pdf_file_name, bbox_inches='tight')
    print(f"Generated {pdf_file_name}")

    plt.show()
    print(f"Generated {file_name}")

# --- 执行生成 ---
# 1. 生成 SR (Success Rate) 图表
draw_rag_comparison_final('SR', 'Figure4-1.png', 'Success Rate (%)')

# 2. 生成 F1 (F1-Score) 图表
draw_rag_comparison_final('F1', 'Figure4-2.png', 'F1-Score (%)')