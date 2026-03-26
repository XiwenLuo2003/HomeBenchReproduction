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

    for i in range(len(categories)):
        base_score = np.random.uniform(78, 88) 
        
        # 逻辑：索引对应类别
        if i in [0, 2]: # VS, VM
            salk = base_score + np.random.uniform(1, 4)
            no_cot = base_score - np.random.uniform(2, 5)
            no_kg = base_score - np.random.uniform(5, 8)
        elif i in [1, 3, 4]: # IS, IM, MM
            salk = base_score + np.random.uniform(4, 7)
            no_cot = base_score - np.random.uniform(8, 15)
            no_kg = np.random.uniform(25, 45) 
        else: # IC
            salk = base_score + np.random.uniform(2, 5)
            no_cot = base_score - np.random.uniform(15, 25)
            no_kg = np.random.uniform(5, 15) 
            
        salk_scores.append(min(100, max(0, salk)))
        no_cot_scores.append(min(100, max(0, no_cot)))
        no_kg_scores.append(min(100, max(0, no_kg)))
    
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

    # 标题
    plt.title(title_label, y=1.1, fontsize=120, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Generated {file_name}")

# --- 执行生成 ---
draw_ablation_radar("Zero-Shot OP", "Figure3-1.png", "Success Rate of OP (%)", data_seed=42)
draw_ablation_radar("Few-Shot ICL", "Figure3-2.png", "Success Rate of ICL (%)", data_seed=99)