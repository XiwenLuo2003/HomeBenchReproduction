import matplotlib.pyplot as plt
import numpy as np

# --- 设置全局学术风格字体 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 26 

def draw_ablation_radar(setting_name, file_name, title_label, data_seed=42):
    """
    绘制消融实验雷达图 (最终完美版：全称标签 + 清晰斑马纹 + 刻度偏移)
    """
    np.random.seed(data_seed)
    
    # 1. 修改为全称标签 (使用 \n 换行以优化布局，去掉缩写)
    categories = [ 'Valid Single', 'Invalid Single', 'Valid Multi', 'Invalid Multi', 'Mix Multi', 'Implicit Context' ]
    N = len(categories)
    
    # 计算角度 (0度, 60度, 120度...)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] 

    # --- 生成模拟数据 (逻辑保持不变) ---
    salk_scores = []
    no_cot_scores = []
    no_kg_scores = []

    
    # --- 生成模拟数据 (删除原来的 for 循环，替换为下面的代码) ---
    
    if "OP" in setting_name:
        # 这里填入 Zero-Shot OP 的真实数据 (顺序: VS, IS, VM, IM, MM, IC)
        salk_scores   = [85.5, 92.1, 80.5, 88.2, 75.4, 48.0] 
        no_cot_scores = [70.2, 55.1, 68.4, 45.3, 50.1, 25.2]
        no_kg_scores  = [68.5, 20.8, 65.2, 15.6, 25.8, 10.5]
        
    else: # 对应 ICL 设置
        # 这里填入 Few-Shot ICL 的真实数据
        salk_scores   = [88.2, 95.5, 83.1, 91.5, 79.2, 55.6]
        no_cot_scores = [75.1, 60.2, 72.5, 50.1, 58.4, 30.5]
        no_kg_scores  = [70.5, 25.1, 68.4, 18.2, 30.1, 15.2]
    
    # 数据闭合
    salk_scores += salk_scores[:1]
    no_cot_scores += no_cot_scores[:1]
    no_kg_scores += no_kg_scores[:1]

    # --- 绘图 ---
    # 大小
    fig, ax = plt.subplots(figsize=(50, 50), subplot_kw=dict(polar=True))

    # 定义配色 (蓝/紫/粉)
    color_salk = '#3DA9E6'    # 蓝
    color_no_tdar = '#9B59B6' # 紫
    color_no_dssd = '#F56FA6' # 粉

    # 2. 【关键修改】绘制清晰的背景色环 (Zebra Stripes)
    # 使用较深的冷灰色，确保在PDF中清晰可见
    yticks = [0, 20, 40, 60, 80, 100]
    for i in range(len(yticks)-1):
        if i % 2 == 0: 
            # 偶数环：填充冷灰色
            ax.fill_between(
                np.linspace(0, 2*np.pi, 100), 
                yticks[i], yticks[i+1], 
                color='#D3D3D3', alpha=0.3, zorder=0 
            )
        else:
            # 奇数环：白色
             ax.fill_between(
                np.linspace(0, 2*np.pi, 100), 
                yticks[i], yticks[i+1], 
                color='white', alpha=1.0, zorder=0
            )

    # 3. 绘制雷达线 (带标记点)
    
    # SALK (Ours)
    ax.plot(angles, salk_scores, 'o-', linewidth=6, markersize=9, 
            label='SALK (Ours)', color=color_salk, zorder=3)
    ax.fill(angles, salk_scores, color=color_salk, alpha=0.15, zorder=3)

    # w/o T-DAR
    ax.plot(angles, no_cot_scores, 'o--', linewidth=6, markersize=9, 
            label='w/o T-DAR', color=color_no_tdar, zorder=2)
    ax.fill(angles, no_cot_scores, color=color_no_tdar, alpha=0.05, zorder=2)

    # w/o D-SSD
    ax.plot(angles, no_kg_scores, 'o:', linewidth=6, markersize=9, 
            label='w/o D-SSD', color=color_no_dssd, zorder=1)
    ax.fill(angles, no_kg_scores, color=color_no_dssd, alpha=0.05, zorder=1)

    # --- 装饰图表 ---
    # 设置 X 轴标签
    ax.set_xticks(angles[:-1])
    ## 雷达图六个方向的标签字体
    ax.set_xticklabels(categories, 
    # fontweight='bold', 
    fontsize=114, color='black')
    ax.tick_params(axis='x', pad=35) # 增加标签距离，防止碰到图

    # 4. 【关键修改】Y轴刻度位置调整
    # 0度是 Valid Single, 60度(2pi/6) 是 Invalid Single
    # 我们将刻度移动到 30度 (两者中间)
    ax.set_rlabel_position(30) 
    # 刻度字体
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], 
               color="black", size=114, fontweight='bold') 
    plt.ylim(0, 100)

    # 网格线 (白色，覆盖在灰色环上会很清楚)
    ax.grid(True, linestyle='-', color='white', linewidth=4.5, alpha=0.6, zorder=1)
    
    # 去掉外圈框线
    ax.spines['polar'].set_visible(False)
    
    # 补一条外圈黑线 (100的位置)，让图看起来有边界感
    ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*100, color='grey', linewidth=3, linestyle='-')

    # 图标示例字体
    # 图例设置
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=3, frameon=True, edgecolor='black', fontsize=120,
              handlelength=2.5) 
    
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=3, frameon=True, edgecolor='black', fontsize=120,
              handlelength=2.5) 
    
    # 【关键修改】加粗图例边框
    # 因为您的图非常大，linewidth=1是看不见的，这里设置为 5
    leg.get_frame().set_linewidth(5.0)

    # 标题
    plt.title(title_label, y=0.9, fontsize=120, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Generated {file_name}")

# --- 执行生成 ---
draw_ablation_radar("Zero-Shot OP", "Figure3-1.png", "Success Rate of OP (%)", data_seed=42)
draw_ablation_radar("Few-Shot ICL", "Figure3-2.png", "Success Rate of ICL (%)", data_seed=99)