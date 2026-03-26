import matplotlib.pyplot as plt
import numpy as np

# --- 设置全局学术风格字体 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# 【关键修改】基础字体放大到 60
plt.rcParams['font.size'] = 60
# 确保 PDF 字体可编辑
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def draw_combined_radar():
    """
    绘制 2x2 组合雷达图：
    - 半径极限最大化 (超大画布 + 负间距)
    - 字体极安 (适配论文缩放)
    - 标题紧贴
    """
    
    # 标签 (全称 + 换行)
    categories = [ 'Valid\nSingle', 'Invalid\nSingle', 'Valid\nMulti', 'Invalid\nMulti', 'Mix\nMulti', 'Implicit\nContext' ]
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] 
    
    # --- 数据准备 ---
    # OP SR
    op_sr_salk    = [55.03, 76.48, 29.8, 0, 0.22, 57.26]
    op_sr_tdar    = [52.32, 42.52, 25.31, 0, 0, 57.14]
    op_sr_dssd    = [49.41, 29.42, 12.65, 0, 0, 55.12]

    # OP P
    op_p_salk     = [54.91, 76.18, 56.3, 56.52, 54.53, 57.09]
    op_p_tdar     = [52.27, 42.35, 58.52, 44.85, 43.28, 56.94]
    op_p_dssd     = [49.38, 29.21, 42.69, 19.76, 29.79, 54.95]

    # ICL SR
    icl_sr_salk   = [74.96, 82.48, 40.00, 0, 0.59, 75.10]
    icl_sr_tdar   = [69.62, 8.85, 26.53, 9.09, 29.81, 74.16]
    icl_sr_dssd   = [64.66, 6.24, 40.00, 0, 0.34, 71.25] 

    # ICL P
    icl_p_salk    = [74.87, 82.37, 72.93, 12.33, 34.50, 75.00]
    icl_p_tdar    = [69.48, 8.81, 58.73, 9.09, 29.81, 74.16]
    icl_p_dssd    = [64.57, 6.20, 70.70, 4.48, 28.28, 70.98]

    # 组合数据配置
    plots_config = [
        ("Success Rate of OP (%)", op_sr_salk, op_sr_tdar, op_sr_dssd),
        ("Precision of OP (%)", op_p_salk, op_p_tdar, op_p_dssd),
        ("Success Rate of ICL (%)", icl_sr_salk, icl_sr_tdar, icl_sr_dssd),
        ("Precision of ICL (%)", icl_p_salk, icl_p_tdar, icl_p_dssd)
    ]

    # --- 关键修改 1：画布尺寸 40x40 (保持大画布) ---
    fig, axs = plt.subplots(2, 2, figsize=(40, 40), subplot_kw=dict(polar=True))
    axs = axs.flatten() 

    color_salk = '#3DA9E6'
    color_no_tdar = '#9B59B6'
    color_no_dssd = '#F56FA6'

    legend_handles = []
    legend_labels = []

    for idx, (title, salk, no_tdar, no_dssd) in enumerate(plots_config):
        ax = axs[idx]
        
        d1 = list(salk) + [salk[0]]
        d2 = list(no_tdar) + [no_tdar[0]]
        d3 = list(no_dssd) + [no_dssd[0]]

        # 绘制背景 (Zebra Stripes)
        yticks = [0, 20, 40, 60, 80, 100]
        for i in range(len(yticks)-1):
            if i % 2 == 0: 
                ax.fill_between(np.linspace(0, 2*np.pi, 100), yticks[i], yticks[i+1], color='#D3D3D3', alpha=0.3, zorder=0)
            else:
                 ax.fill_between(np.linspace(0, 2*np.pi, 100), yticks[i], yticks[i+1], color='white', alpha=1.0, zorder=0)

        # 绘制数据线 (【关键修改】线宽 10.0, 点大小 30)
        l1, = ax.plot(angles, d1, 'o-', linewidth=10.0, markersize=30, label='SALK (Ours)', color=color_salk, zorder=3)
        ax.fill(angles, d1, color=color_salk, alpha=0.15, zorder=3)

        l2, = ax.plot(angles, d2, 'o--', linewidth=10.0, markersize=30, label='w/o T-DAR', color=color_no_tdar, zorder=2)
        ax.fill(angles, d2, color=color_no_tdar, alpha=0.05, zorder=2)

        l3, = ax.plot(angles, d3, 'o:', linewidth=10.0, markersize=30, label='w/o D-SSD', color=color_no_dssd, zorder=1)
        ax.fill(angles, d3, color=color_no_dssd, alpha=0.05, zorder=1)
        
        if idx == 0:
            legend_handles = [l1, l2, l3]
            legend_labels = ['SALK (Ours)', 'w/o T-DAR', 'w/o D-SSD']

        # --- 装饰每个子图 ---
        ax.set_xticks(angles[:-1])
        # 标签字体 60，距离 50
        ax.set_xticklabels(categories, fontsize=60, color='black') 
        ax.tick_params(axis='x', pad=50) 

        ax.set_rlabel_position(30) 
        # Y轴刻度字体 45
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], color="black", size=45, fontweight='bold')
        ax.set_ylim(0, 100)

        # 网格线宽 6.0
        ax.grid(True, linestyle='-', color='white', linewidth=6.0, alpha=0.6, zorder=1)
        ax.spines['polar'].set_visible(False)
        ax.plot(np.linspace(0, 2*np.pi, 100), np.ones(100)*100, color='grey', linewidth=6.0, linestyle='-')

        # 子图标题 (【关键修改】位置下移紧贴 y=1.10, 字体 70)
        ax.set_title(title, y=1.10, fontsize=70, fontweight='bold', pad=30)

    # --- 关键修改 2：使用负间距 (Negative Spacing) ---
    # wspace=-0.15: 这是一个非常激进的负值，能让左右两个雷达图的空白边框重叠，从而让视觉中心靠得更近
    # hspace=0.30: 垂直间距保持适中
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.12, wspace=-0.15, hspace=0.30)
    
    # 底部图例 (字体 65)
    leg = fig.legend(legend_handles, legend_labels, loc='lower center', 
                     ncol=3, frameon=True, edgecolor='black', fontsize=65,
                     bbox_to_anchor=(0.5, 0.015), handlelength=2.5)
    leg.get_frame().set_linewidth(6.0)
    
    # 保存
    file_name = "Figure3_Combined.png"
    pdf_file = file_name.replace('.png', '.pdf')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()
    print(f"Generated {file_name} and {pdf_file}")

# 执行
draw_combined_radar()