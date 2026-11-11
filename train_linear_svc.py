import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

train_df = pd.read_csv("results/train.csv")
test_df = pd.read_csv("results/test.csv")

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2, max_features=15000)
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

model = LinearSVC(C=1.0, random_state=42)
model.fit(X_train, train_df['author'])
predictions = model.predict(X_test)

pd.DataFrame({"pred_linear_svc": predictions}).to_csv("results/preds_linear_svc.csv", index=False)

metrics = {
    "accuracy": float(accuracy_score(test_df['author'], predictions)),
    "macro_f1": float(f1_score(test_df['author'], predictions, average='macro'))
}

with open("results/metrics_linear_svc.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"[LinearSVC] Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")