import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

test_df = pd.read_csv("results/test.csv")
true_authors = test_df['author'].values

models_config = [
    ("Naive Bayes", "results/metrics_naive_bayes.json", "results/preds_naive_bayes.csv", "pred_naive_bayes"),
    ("LinearSVC", "results/metrics_linear_svc.json", "results/preds_linear_svc.csv", "pred_linear_svc"),
    ("Ukr-RoBERTa", "results/metrics_transformer.json", "results/preds_transformer.csv", "pred_transformer")
]

summary_data = {}
all_preds = {"true_author": true_authors}

for name, metric_path, pred_path, col_name in models_config:
    with open(metric_path, "r", encoding="utf-8") as f:
        summary_data[name] = json.load(f)
    preds = pd.read_csv(pred_path)[col_name].values
    all_preds[col_name] = preds

summary_df = pd.DataFrame.from_dict(summary_data, orient="index")
summary_df.columns = ["Accuracy", "Macro-F1"]

print("\n" + "="*45)
print("ПІДСУМКОВА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
print("="*45)
print(summary_df.to_string())
print("="*45)

summary_df.to_csv("results/summary_metrics.csv")

# Графік 1: Порівняння метрик
fig, ax = plt.subplots(figsize=(8, 5))
summary_df.plot(kind="bar", ax=ax, rot=0, colormap="viridis", edgecolor="black")
ax.set_title("Порівняння ефективності моделей за рівнем складності", fontsize=12, pad=12)
ax.set_ylabel("Значення метрики", fontsize=10)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=9, xytext=(0, 4), textcoords='offset points')

plt.tight_layout()
plt.savefig("results/metrics_comparison.png", dpi=300)
plt.close()

# Графік 2: Матриці плутанини
author_classes = sorted(list(set(true_authors)))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, _, _, col_name) in enumerate(models_config):
    cm = confusion_matrix(true_authors, all_preds[col_name], labels=author_classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=author_classes,
                yticklabels=author_classes, ax=axes[idx], cbar=False)
    axes[idx].set_title(f"Матриця: {name}", fontsize=11)
    axes[idx].set_xlabel("Передбачено", fontsize=9)
    axes[idx].set_ylabel("Справжній клас", fontsize=9)
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("results/confusion_matrices.png", dpi=300)
plt.close()

print("\nГрафіки та підсумковий звіт збережено в папку results/:")
print("- results/summary_metrics.csv")
print("- results/metrics_comparison.png")
print("- results/confusion_matrices.png")