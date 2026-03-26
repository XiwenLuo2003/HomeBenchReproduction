import matplotlib.pyplot as plt
import numpy as np

# --- 设置全局学术风格字体 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# 【关键修改】基础字体统一调整为 28 (标题大小)
plt.rcParams['font.size'] = 28
# 确保 PDF 字体可编辑
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def draw_ablation_radar_optimized(file_name, title_label, salk_data, no_tdar_data, no_dssd_data):
    """
    绘制消融实验雷达图 (Overleaf 优化版：全大字体)
    """
    
    # 标签 (全称 + 换行)
    categories = [ 'Valid\nSingle', 'Invalid\nSingle', 'Valid\nMulti', 'Invalid\nMulti', 'Mix\nMulti', 'Implicit\nContext' ]
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] 
    
    # 数据闭合
    salk_data = list(salk_data) + [salk_data[0]]
    no_tdar_data = list(no_tdar_data) + [no_tdar_data[0]]
    no_dssd_data = list(no_dssd_data) + [no_dssd_data[0]]

    # --- 关键修改 1：画布尺寸设为 12x12 (原 10x10) ---
    # 画布扩大 1.2 倍，在字号不变的情况下，雷达图半径会自动扩大以填充空间
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    color_salk = '#3DA9E6'
    color_no_tdar = '#9B59B6'
    color_no_dssd = '#F56FA6'

    # 绘制背景 (Zebra Stripes)
    yticks = [0, 20, 40, 60, 80, 100]
    for i in range(len(yticks)-1):
        if i % 2 == 0: 
            ax.fill_between(np.linspace(0, 2*np.pi, 100), yticks[i], yticks[i+1], color='#D3D3D3', alpha=0.3, zorder=0)
        else:
             ax.fill_between(np.linspace(0, 2*np.pi, 100), yticks[i], yticks[i+1], color='white', alpha=1.0, zorder=0)

    # --- 线宽和点大小 ---
    # 保持适中：线宽 3.0, 点大小 10
    
    # SALK (Ours)
    ax.plot(angles, salk_data, 'o-', linewidth=3.0, markersize=10, 
            label='SALK (Ours)', color=color_salk, zorder=3)
    ax.fill(angles, salk_data, color=color_salk, alpha=0.15, zorder=3)

    # w/o T-DAR
    ax.plot(angles, no_tdar_data, 'o--', linewidth=3.0, markersize=10, 
            label='w/o T-DAR', color=color_no_tdar, zorder=2)
    ax.fill(angles, no_tdar_data, color=color_no_tdar, alpha=0.05, zorder=2)

    # w/o D-SSD
    ax.plot(angles, no_dssd_data, 'o:', linewidth=3.0, markersize=10, 
            label='w/o D-SSD', color=color_no_dssd, zorder=1)
    ax.fill(angles, no_dssd_data, color=color_no_dssd, alpha=0.05, zorder=1)

    # --- 装饰图表 ---
    ax.set_xticks(angles[:-1])
    
    # --- 字体大小优化 (全部统一为 28) ---
    # 标签字体 28
    ax.set_xticklabels(categories, fontsize=28, color='black') 
    # 增加标签距离 (pad 15 -> 20)
    ax.tick_params(axis='x', pad=20) 

    # Y轴刻度
    ax.set_rlabel_position(30) 
    # 刻度数字字体 28
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], 
               color="black", size=28, fontweight='bold') 
    plt.ylim(0, 100)

    # 网格线宽
    ax.grid(True, linestyle='-', color='white', linewidth=2.0, alpha=0.6, zorder=1)
    ax.spines['polar'].set_visible(False)
    
    # 边界线宽
    ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*100, color='grey', linewidth=2.0, linestyle='-')

    # 图例字体 28
    # 位置下移 (-0.10 -> -0.15) 避免遮挡
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=3, frameon=True, edgecolor='black', fontsize=28,
              handlelength=2.0) 
    leg.get_frame().set_linewidth(2.0)

    # 标题字体 28
    # 位置上移 (1.10 -> 1.15)
    plt.title(title_label, y=1.15, fontsize=28, fontweight='bold')
    
    # 使用 tight_layout 自动调整布局，防止文字被切
    plt.tight_layout()
    
    # --- 保存 ---
    pdf_file = file_name.replace('.png', '.pdf')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()
    print(f"Generated {file_name} and {pdf_file}")

# --- 真实数据录入 ---

# 1. OP SR
op_sr_salk    = [55.03, 76.48, 29.8, 0, 0.22, 57.26]
op_sr_tdar    = [52.32, 42.52, 25.31, 0, 0, 57.14]
op_sr_dssd    = [49.41, 29.42, 12.65, 0, 0, 55.12]
draw_ablation_radar_optimized("Figure3-1_OP_SR.png", "Success Rate of OP (%)", op_sr_salk, op_sr_tdar, op_sr_dssd)

# 2. OP P
op_p_salk     = [54.91, 76.18, 56.3, 56.52, 54.53, 57.09]
op_p_tdar     = [52.27, 42.35, 58.52, 44.85, 43.28, 56.94]
op_p_dssd     = [49.38, 29.21, 42.69, 19.76, 29.79, 54.95]
draw_ablation_radar_optimized("Figure3-2_OP_P.png", "Precision of OP (%)", op_p_salk, op_p_tdar, op_p_dssd)

# 3. ICL SR
icl_sr_salk   = [74.96, 82.48, 40.00, 0, 0.59, 75.10]
icl_sr_tdar   = [69.62, 8.85, 26.53, 9.09, 29.81, 74.16]
icl_sr_dssd   = [64.66, 6.24, 40.00, 0, 0.34, 71.25] 
draw_ablation_radar_optimized("Figure3-3_ICL_SR.png", "Success Rate of ICL (%)", icl_sr_salk, icl_sr_tdar, icl_sr_dssd)

# 4. ICL P
icl_p_salk    = [74.87, 82.37, 72.93, 12.33, 34.50, 75.00]
icl_p_tdar    = [69.48, 8.81, 58.73, 9.09, 29.81, 74.16]
icl_p_dssd    = [64.57, 6.20, 70.70, 4.48, 28.28, 70.98]
draw_ablation_radar_optimized("Figure3-4_ICL_P.png", "Precision of ICL (%)", icl_p_salk, icl_p_tdar, icl_p_dssd)