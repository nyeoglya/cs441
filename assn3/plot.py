import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

path = sys.argv[1] # 'training_log_aug2'

# 데이터
epochs = list(range(1, 31))
train_loss = [2.4620, 2.0487, 1.8638, 1.7271, 1.6233, 1.5507, 1.4977, 1.4495, 1.4191, 1.3938, 1.3760, 1.3556, 1.3369, 1.3215, 1.3097, 1.2921, 1.2753, 1.2531, 1.2370, 1.2094, 1.1853, 1.1461, 1.1070, 1.0541, 0.9991, 0.9218, 0.8415, 0.7622, 0.6993, 0.6724]
val_loss = [2.2120, 2.0237, 2.1347, 1.7795, 1.9243, 1.5420, 1.6984, 1.6038, 1.5511, 1.4843, 1.4614, 1.6033, 1.4413, 1.5130, 1.3831, 1.4535, 1.4549, 1.3574, 1.3276, 1.4023, 1.3347, 1.2165, 1.2061, 1.1686, 1.1454, 1.0685, 1.0055, 0.9610, 0.9436, 0.9339]
train_acc = [0.2246, 0.3955, 0.4711, 0.5249, 0.5656, 0.5958, 0.6159, 0.6358, 0.6496, 0.6595, 0.6670, 0.6753, 0.6817, 0.6891, 0.6919, 0.7036, 0.7072, 0.7175, 0.7244, 0.7332, 0.7436, 0.7604, 0.7769, 0.8010, 0.8203, 0.8558, 0.8886, 0.9249, 0.9531, 0.9659]
val_acc = [0.3221, 0.4084, 0.4037, 0.5028, 0.4781, 0.5903, 0.5360, 0.5913, 0.5949, 0.6274, 0.6270, 0.5753, 0.6377, 0.6240, 0.6709, 0.6366, 0.6353, 0.6726, 0.6877, 0.6576, 0.6864, 0.7324, 0.7327, 0.7467, 0.7617, 0.7889, 0.8214, 0.8362, 0.8500, 0.8502]

best_idx = int(np.argmax(val_acc))
best_epoch = epochs[best_idx]
best_val_acc = val_acc[best_idx]

# 디자인 설정
sns.set_theme(style="white", context="talk")
colors = ["#F5D3A2", "#F5A7A2", "#A4BEF5", "#D8A4F5" ]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Loss Curve
sns.lineplot(x=epochs, y=train_loss, ax=axes[0], color=colors[0], label="Train", linewidth=3, marker='o', markersize=8)
sns.lineplot(x=epochs, y=val_loss, ax=axes[0], color=colors[1], label="Validation", linewidth=3, marker='s', markersize=8)

# Accuracy Curve
sns.lineplot(x=epochs, y=train_acc, ax=axes[1], color=colors[2], label="Train", linewidth=3, marker='o', markersize=8)
sns.lineplot(x=epochs, y=val_acc, ax=axes[1], color=colors[3], label="Validation", linewidth=3, marker='s', markersize=8)

# 강조 포인트 및 디테일
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_xlabel("Epochs")

axes[0].set_title("Learning Loss", fontsize=20, pad=20)
axes[1].set_title("Learning Accuracy", fontsize=20, pad=20)

# Best Accuracy 하이라이트
axes[1].plot(best_epoch, best_val_acc, marker='*', markersize=20, color="#E9C46A", markeredgecolor="black")
axes[1].annotate(
    f'Best: {best_val_acc:.2%}',
    xy=(best_epoch, best_val_acc),
    xytext=(best_epoch - 3, best_val_acc - 0.05),
    fontsize=16, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
)

plt.tight_layout()
# plt.show()
plt.savefig(path + '.png')
