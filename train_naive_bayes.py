import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

train_df = pd.read_csv("results/train.csv")
test_df = pd.read_csv("results/test.csv")

vectorizer = TfidfVectorizer(max_features=4000, lowercase=True)
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

model = MultinomialNB()
model.fit(X_train, train_df['author'])
predictions = model.predict(X_test)

pd.DataFrame({"pred_naive_bayes": predictions}).to_csv("results/preds_naive_bayes.csv", index=False)

metrics = {
    "accuracy": float(accuracy_score(test_df['author'], predictions)),
    "macro_f1": float(f1_score(test_df['author'], predictions, average='macro'))
}

with open("results/metrics_naive_bayes.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"[Naive Bayes] Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")