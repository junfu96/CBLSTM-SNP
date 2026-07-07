import numpy as np
import os
import matplotlib.pyplot as plt

DATASETS = ["sp500", "milk", "closings", "lake", "MG"]

# =========================
# 绘图函数（优化版）
# =========================
def plot_snp_comparison(dataset):

    base = f"./results/{dataset}/"

    lstm_snp = np.loadtxt(base + "4gate.csv", delimiter=',')
    cblstm_snp = np.loadtxt(base + "5gate_lambda.csv", delimiter=',')

    gt = np.loadtxt(base + "gt.csv", delimiter=',')

    # ===== 误差 =====
    err_lstm = np.abs(gt - lstm_snp)
    err_cblstm = np.abs(gt - cblstm_snp)

    # 差值（关键）
    err_diff = err_lstm - err_cblstm

    # ===== 风格 =====
    plt.style.use('grayscale')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # =========================
    # (a) Prediction（只画两个模型）
    # =========================
    axes[0].plot(lstm_snp,
                 linestyle='--',
                 marker='o',
                 markersize=2,
                 linewidth=1.2,
                 label='LSTM-SNP')

    axes[0].plot(cblstm_snp,
                 linestyle='-',
                 marker='s',
                 markersize=2,
                 linewidth=1.2,
                 label='CBLSTM-SNP')

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend(fontsize=8)

    axes[0].text(0.02, 0.95, "(a)", transform=axes[0].transAxes)

    # =========================
    # (b) Error（增强区分）
    # =========================
    axes[1].plot(err_lstm,
                 linestyle='--',
                 marker='o',
                 markersize=2,
                 linewidth=1,
                 label='LSTM-SNP Error')

    axes[1].plot(err_cblstm,
                 linestyle='-',
                 marker='s',
                 markersize=2,
                 linewidth=1,
                 label='CBLSTM-SNP Error')

    # 👉 差值曲线（关键增强）
    axes[1].plot(err_diff,
                 linestyle=':',
                 linewidth=1.5,
                 label='Error Difference')

    axes[1].axhline(0, linestyle='-', linewidth=1)

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Error")
    axes[1].legend(fontsize=8)

    axes[1].text(0.02, 0.95, "(b)", transform=axes[1].transAxes)

    # =========================
    # 统一风格
    # =========================
    for ax in axes:
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout()

    # ===== 保存 =====
    os.makedirs("Compare", exist_ok=True)
    save_path = f"Compare/{dataset}_snp_compare.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Saved] {save_path}")


# =========================
# 主函数
# =========================
if __name__ == "__main__":

    for d in DATASETS:
        plot_snp_comparison(d)

    print("All optimized SNP comparison figures generated.")