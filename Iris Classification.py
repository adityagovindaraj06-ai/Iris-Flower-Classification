
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sns.set_theme(style="whitegrid")

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = pd.Categorical.from_codes(
    iris.target,
    iris.target_names
)

print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nClass balance:\n", df["species"].value_counts())
print("\nSummary stats:\n", df.describe())

pairplot = sns.pairplot(
    df,
    hue="species",
    diag_kind="hist"
)

pairplot.fig.suptitle(
    "Iris Feature Pairplot",
    y=1.02
)

pairplot.savefig(
    "./iris_pairplot.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8)
)

for ax, col in zip(
    axes.ravel(),
    iris.feature_names
):
    sns.histplot(
        data=df,
        x=col,
        hue="species",
        kde=True,
        ax=ax,
        legend=(col == iris.feature_names[0])
    )
    ax.set_title(col)

fig.tight_layout()

fig.savefig(
    "./iris_histograms.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

X = df[iris.feature_names]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    f"\nTrain size: {X_train.shape[0]}, "
    f"Test size: {X_test.shape[0]}"
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

results = {}

for name, model in models.items():

    if name == "Decision Tree":
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

    acc = accuracy_score(
        y_test,
        preds
    )

    prec = precision_score(
        y_test,
        preds,
        average="macro"
    )

    rec = recall_score(
        y_test,
        preds,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        preds,
        average="macro"
    )

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    print(
        f"\n{'=' * 50}\n"
        f"{name}\n"
        f"{'=' * 50}"
    )

    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")

    print(
        "\nClassification report:\n",
        classification_report(y_test, preds)
    )

    cm = confusion_matrix(
        y_test,
        preds,
        labels=iris.target_names
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.title(
        f"Confusion Matrix — {name}"
    )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    fname = (
        f"./confusion_matrix_"
        f"{name.replace(' ', '_').lower()}.png"
    )

    plt.savefig(
        fname,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

results_df = pd.DataFrame(
    results
).T.sort_values(
    "accuracy",
    ascending=False
)

print(
    "\n\nModel comparison:\n",
    results_df
)

results_df.to_csv(
    "./model_comparison.csv"
)

print(
    "\nDone. Plots and comparison CSV saved to ./"
	)