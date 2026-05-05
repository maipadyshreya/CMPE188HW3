#import libraries
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


from gensim.models import Word2Vec

def tokenize_texts(texts):
    return [str(text).lower().split() for text in texts]

def avg_word2vec(texts, model, vector_size=100):
    vectors = []

    for words in tokenize_texts(texts):
        word_vecs = [model.wv[word] for word in words if word in model.wv]

        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(vector_size))

    return np.array(vectors)

# Load dataset
df = pd.read_csv("train_with_task_type.csv")

print(df.head())
print(df.columns)

# Select text and label columns
LABEL_COL = "task_type"
possible_text_cols = ["prompt", "question", "text", "input", "problem", "response"]

TEXT_COL = None
for col in possible_text_cols:
    if col in df.columns:
        TEXT_COL = col
        break

if TEXT_COL is None:
    raise ValueError("No valid text column found. Check dataset columns.")

print("Using text column:", TEXT_COL)
print("Using label column:", LABEL_COL)

df = df[[TEXT_COL, LABEL_COL]].dropna()

X = df[TEXT_COL].astype(str)
y = df[LABEL_COL].astype(str)

print(y.value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# TF-IDF pipelines
pipelines = {
    "TF-IDF + Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
        ("model", MultinomialNB())
    ]),

    "TF-IDF + SVM": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))),
        ("model", LinearSVC())
    ]),

    "TF-IDF + KNN": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "TF-IDF + Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),

    "TF-IDF + Neural Network": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
        ("model", MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42))
    ])
}


results = []

# Train and evaluate TF-IDF models
for name, pipeline in pipelines.items():
    print(f"\nTraining: {name}")

    train_start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - train_start

    inference_start = time.time()
    y_pred = pipeline.predict(X_test)
    inference_time = time.time() - inference_start

    results.append({
        "Pipeline": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Training Time": train_time,
        "Inference Time": inference_time
    })

    print(classification_report(y_test, y_pred, zero_division=0))


# Word2Vec Embeddings + SVM
print("\nTraining: Word2Vec Embeddings + SVM")

train_start = time.time()

# Train Word2Vec model
tokenized_train = tokenize_texts(X_train)

w2v_model = Word2Vec(
    sentences=tokenized_train,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    seed=42
)

# Convert text → vectors
X_train_emb = avg_word2vec(X_train, w2v_model)
X_test_emb = avg_word2vec(X_test, w2v_model)

# Train classifier
embedding_model = LinearSVC(random_state=42)
embedding_model.fit(X_train_emb, y_train)

train_time = time.time() - train_start

# Evaluate
inference_start = time.time()
y_pred = embedding_model.predict(X_test_emb)
inference_time = time.time() - inference_start

results.append({
    "Pipeline": "Word2Vec + SVM",
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    "Training Time": train_time,
    "Inference Time": inference_time
})

print(classification_report(y_test, y_pred, zero_division=0))


# Final results table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1 Score", ascending=False)

print("\nFinal Model Comparison:")
print(results_df)

results_df.to_csv("model_results.csv", index=False)


# Visualization 1: F1 Score comparison
plt.figure(figsize=(10, 5))
plt.bar(results_df["Pipeline"], results_df["F1 Score"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1 Score")
plt.title("Model F1 Score Comparison")
plt.tight_layout()
plt.savefig("f1_score_comparison.png")
plt.show()


# Visualization 2: Training time comparison
plt.figure(figsize=(10, 5))
plt.bar(results_df["Pipeline"], results_df["Training Time"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Training Time (seconds)")
plt.title("Model Training Time Comparison")
plt.tight_layout()
plt.savefig("training_time_comparison.png")
plt.show()


# Visualization 3: Performance vs speed
plt.figure(figsize=(8, 5))
plt.scatter(results_df["Training Time"], results_df["F1 Score"])

for i in range(len(results_df)):
    plt.text(
        results_df["Training Time"].iloc[i],
        results_df["F1 Score"].iloc[i],
        results_df["Pipeline"].iloc[i],
        fontsize=8
    )

plt.xlabel("Training Time (seconds)")
plt.ylabel("F1 Score")
plt.title("Performance vs Training Speed")
plt.tight_layout()
plt.savefig("speed_vs_performance.png")
plt.show()