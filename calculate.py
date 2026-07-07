import numpy as np
import os
import matplotlib.pyplot as plt

# 1.数据集
DATASETS = ["sp500", "milk", "closings", "lake", "MG"]

# =========================
# 论文级绘图
# =========================
def plot_paper_figure(dataset):

    base = f"./results/{dataset}/"

    gt = np.loadtxt(base + "gt.csv", delimiter=',')
    lstm = np.loadtxt(base + "lstm.csv", delimiter=',')
    g4 = np.loadtxt(base + "4gate.csv", delimiter=',')
    g5l = np.loadtxt(base + "5gate_lambda.csv", delimiter=',')

    # ===== 误差计算 =====
    err_lstm = np.abs(gt - lstm)
    err_g4 = np.abs(gt - g4)
    err_g5l = np.abs(gt - g5l)

    # ===== 风格 =====
    plt.style.use('grayscale')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # =========================
    # (a) 原始 vs 预测
    # =========================
    axes[0].plot(gt, 'ko--', markersize=3, linewidth=1.2,
                 label='original data')

    axes[0].plot(g5l, 'k-', linewidth=2,
                 label='predicted data')

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend(fontsize=8)

    axes[0].text(0.02, 0.95, "(a)", transform=axes[0].transAxes, fontsize=10)

    # =========================
    # (b) 误差差值图（关键改进）
    # =========================
    axes[1].plot(err_lstm - err_g5l,
                 linestyle='--', linewidth=1.5,
                 label='LSTM - Proposed')

    axes[1].plot(err_g4 - err_g5l,
                 linestyle='-.', linewidth=1.5,
                 label='4-gate - Proposed')

    axes[1].axhline(0, linestyle='-', linewidth=1)

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Error Difference")
    axes[1].legend(fontsize=8)

    axes[1].text(0.02, 0.95, "(b)", transform=axes[1].transAxes, fontsize=10)

    # =========================
    # (c) Boxplot（统计对比）
    # =========================
    axes[2].boxplot(
        [err_lstm, err_g4, err_g5l],
        labels=['LSTM', '4-gate', '5-gate+λ']
    )

    axes[2].set_ylabel("Absolute Error")
    axes[2].text(0.02, 0.95, "(c)", transform=axes[2].transAxes, fontsize=10)

    # =========================
    # 统一风格
    # =========================
    for ax in axes:
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout()

    # ===== 保存 =====
    save_path = f"Figure/{dataset}_final_figure.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Saved] {save_path}")


# =========================
# 批量生成
# =========================
if __name__ == "__main__":

    os.makedirs("Figure", exist_ok=True)

    for d in DATASETS:
        plot_paper_figure(d)

    print("All paper figures generated.")