import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

train_loss = [2.1151, 1.9954, 1.8903, 1.7729, 1.7836, 1.7067, 1.6613,
              1.5961, 1.5800, 1.5028, 1.4904, 1.4626, 1.4702, 1.3938,
              1.4347, 1.4408, 1.3588, 1.3645, 1.3955, 1.3699, 1.3953,
              1.3450, 1.3233, 1.3931, 1.3314, 1.3625, 1.3691, 1.3363,
              1.3936, 1.2922, 1.3480, 1.3228, 1.3201, 1.3670, 1.3035,
              1.3373, 1.3239, 1.2798, 1.2923, 1.3711, 1.3049, 1.2944,
              1.3067, 1.3084, 1.3113, 1.2984, 1.2827, 1.2515, 1.2713, 1.2596]

val_loss = [3.4701, 3.3563, 3.2845, 3.2379, 3.2037, 3.1678, 3.1168,
            3.0808, 3.0396, 2.9917, 2.9500, 2.9115, 2.8790, 2.8540,
            2.8397, 2.8217, 2.8098, 2.7992, 2.7945, 2.7874, 2.7803,
            2.7776, 2.7720, 2.7666, 2.7620, 2.7594, 2.7615, 2.7590,
            2.7597, 2.7553, 2.7543, 2.7520, 2.7489, 2.7469, 2.7452,
            2.7452, 2.7460, 2.7463, 2.7446, 2.7439, 2.7435, 2.7402,
            2.7384, 2.7372, 2.7377, 2.7367, 2.7335, 2.7311, 2.7316, 2.7315]

os.makedirs("outputs/figures", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_loss, label="train", linewidth=2, color="steelblue")
axes[0].plot(val_loss,   label="val",   linewidth=2, color="coral")
axes[0].axhline(y=5.1528, color="gray", linestyle="--",
                linewidth=1, label="naive baseline (5.15)")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("MSE (normalized space)")
axes[0].set_title("PatchTST loss curves -- 50 epochs")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(val_loss, linewidth=2, color="coral", label="val loss")
axes[1].axhline(y=5.1528, color="gray", linestyle="--",
                linewidth=1, label="naive baseline (5.15)")
axes[1].axhline(y=2.7311, color="green", linestyle="--",
                linewidth=1, label="best val (2.73)")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("MSE (normalized space)")
axes[1].set_title("Val loss convergence")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("PatchTST from scratch -- synthetic sales data\n"
             "47% improvement over naive baseline", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/figures/loss_curve_epoch50.png", dpi=150)
print("Saved to outputs/figures/loss_curve_epoch50.png")
