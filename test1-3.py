import matplotlib.pyplot as plt
import numpy as np

# --- 设置全局学术风格字体 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 26 

def draw_ablation_radar_real(file_name, title_label, salk_data, no_tdar_data, no_dssd_data):
    """
    绘制消融实验雷达图 (加粗版：线条加粗一倍，适配超大画布)
    """
    
    # 1. 全称标签
    categories = [ 'Valid Single', 'Invalid Single', 'Valid Multi', 'Invalid Multi', 'Mix Multi', 'Implicit Context' ]
    N = len(categories)
    
    # 计算角度 (闭合)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] 
    
    # 2. 数据闭合
    salk_data = list(salk_data) + [salk_data[0]]
    no_tdar_data = list(no_tdar_data) + [no_tdar_data[0]]
    no_dssd_data = list(no_dssd_data) + [no_dssd_data[0]]

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(50, 50), subplot_kw=dict(polar=True))

    # 定义配色
    color_salk = '#3DA9E6'    # 蓝
    color_no_tdar = '#9B59B6' # 紫
    color_no_dssd = '#F56FA6' # 粉

    # 3. 绘制斑马纹背景
    yticks = [0, 20, 40, 60, 80, 100]
    for i in range(len(yticks)-1):
        if i % 2 == 0: 
            ax.fill_between(
                np.linspace(0, 2*np.pi, 100), 
                yticks[i], yticks[i+1], 
                color='#D3D3D3', alpha=0.3, zorder=0 
            )
        else:
             ax.fill_between(
                np.linspace(0, 2*np.pi, 100), 
                yticks[i], yticks[i+1], 
                color='white', alpha=1.0, zorder=0
            )

    # 4. 绘制数据线 (【关键修改】linewidth 加倍，markersize 增大)
    
    # SALK (Ours)
    ax.plot(angles, salk_data, 'o-', linewidth=12, markersize=24, 
            label='SALK (Ours)', color=color_salk, zorder=3)
    ax.fill(angles, salk_data, color=color_salk, alpha=0.15, zorder=3)

    # w/o T-DAR
    ax.plot(angles, no_tdar_data, 'o--', linewidth=12, markersize=24, 
            label='w/o T-DAR', color=color_no_tdar, zorder=2)
    ax.fill(angles, no_tdar_data, color=color_no_tdar, alpha=0.05, zorder=2)

    # w/o D-SSD
    ax.plot(angles, no_dssd_data, 'o:', linewidth=12, markersize=24, 
            label='w/o D-SSD', color=color_no_dssd, zorder=1)
    ax.fill(angles, no_dssd_data, color=color_no_dssd, alpha=0.05, zorder=1)

    # --- 装饰图表 ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=114, color='black') 
    ax.tick_params(axis='x', pad=35) 

    # Y轴刻度
    ax.set_rlabel_position(30) 
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], 
               color="black", size=114, fontweight='bold') 
    plt.ylim(0, 100)

    # 网格线 (【关键修改】linewidth 加粗到 9)
    ax.grid(True, linestyle='-', color='white', linewidth=9.0, alpha=0.6, zorder=1)
    ax.spines['polar'].set_visible(False)
    
    # 外圈边界线 (【关键修改】linewidth 加粗到 6)
    ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*100, color='grey', linewidth=6.0, linestyle='-')

    # 图例 (边框加粗)
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0), 
              ncol=3, frameon=True, edgecolor='black', fontsize=120,
              handlelength=2.5) 
    leg.get_frame().set_linewidth(6.0)

    # 标题
    plt.title(title_label, y=1.05, fontsize=120, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Generated {file_name}")

# --- 真实数据录入与执行 ---

# ================= 1. OP 设置 =================

# 1-1. Success Rate (SR) of OP
op_sr_salk    = [55.03, 76.48, 29.8, 0, 0.22, 57.26]
op_sr_tdar    = [52.32, 42.52, 25.31, 0, 0, 57.14]
op_sr_dssd    = [49.41, 29.42, 12.65, 0, 0, 55.12]

draw_ablation_radar_real(
    "Figure3-1_OP_SR.png", 
    "Success Rate of OP (%)", 
    op_sr_salk, op_sr_tdar, op_sr_dssd
)

# 1-2. Precision (P) of OP
op_p_salk     = [54.91, 76.18, 56.3, 56.52, 54.53, 57.09]
op_p_tdar     = [52.27, 42.35, 58.52, 44.85, 43.28, 56.94]
op_p_dssd     = [49.38, 29.21, 42.69, 19.76, 29.79, 54.95]

draw_ablation_radar_real(
    "Figure3-2_OP_P.png", 
    "Precision of OP (%)", 
    op_p_salk, op_p_tdar, op_p_dssd
)

# ================= 2. ICL 设置 =================

# 2-1. Success Rate (SR) of ICL
icl_sr_salk   = [74.96, 82.48, 40.00, 0, 0.59, 75.10]
icl_sr_tdar   = [69.62, 8.85, 26.53, 9.09, 29.81, 74.16]
icl_sr_dssd   = [64.66, 6.24, 40.00, 0, 0.34, 71.25] 

draw_ablation_radar_real(
    "Figure3-3_ICL_SR.png", 
    "Success Rate of ICL (%)", 
    icl_sr_salk, icl_sr_tdar, icl_sr_dssd
)

# 2-2. Precision (P) of ICL
icl_p_salk    = [74.87, 82.37, 72.93, 12.33, 34.50, 75.00]
icl_p_tdar    = [69.48, 8.81, 58.73, 9.09, 29.81, 74.16]
icl_p_dssd    = [64.57, 6.20, 70.70, 4.48, 28.28, 70.98]

draw_ablation_radar_real(
    "Figure3-4_ICL_P.png", 
    "Precision of ICL (%)", 
    icl_p_salk, icl_p_tdar, icl_p_dssd
)