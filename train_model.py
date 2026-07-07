import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import ml
from db import load_tweets

REPORTS_DIR = "reports"


def evaluate_label(label_name, y_true, y_pred):
    report = classification_report(y_true, y_pred, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    print(f"\n=== Évaluation : {label_name} ===")
    print(report)
    print("Matrice de confusion :")
    print(matrix)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[f"non {label_name}", label_name],
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Matrice de confusion - {label_name}")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/confusion_matrix_{label_name}.png")
    plt.close()

    return report, matrix


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_tweets()
    if len(df) < 10:
        raise SystemExit(
            "Pas assez de tweets en base pour entraîner un modèle (minimum 10). "
            "Lance d'abord `python seed_data.py`."
        )

    (
        texts_train, texts_test,
        pos_train, pos_test,
        neg_train, neg_test,
    ) = train_test_split(
        df["text"], df["positive"], df["negative"],
        test_size=0.25, random_state=42,
    )

    vectorizer, positive_model, negative_model = ml.train(texts_train, pos_train, neg_train)

    cleaned_test = [ml.clean_text(t) for t in texts_test]
    X_test = vectorizer.transform(cleaned_test)
    positive_pred = positive_model.predict(X_test)
    negative_pred = negative_model.predict(X_test)

    positive_report, _ = evaluate_label("positive", pos_test, positive_pred)
    negative_report, _ = evaluate_label("negative", neg_test, negative_pred)

    ml.save_artifacts(vectorizer, positive_model, negative_model)

    with open(f"{REPORTS_DIR}/last_training.txt", "w", encoding="utf-8") as f:
        f.write(f"Entraînement du {datetime.now().isoformat()}\n")
        f.write(f"Nombre de tweets utilisés : {len(df)}\n\n")
        f.write("=== Rapport positive ===\n")
        f.write(positive_report)
        f.write("\n=== Rapport negative ===\n")
        f.write(negative_report)

    print(f"\nModèles sauvegardés dans `{ml.MODEL_DIR}/`.")
    print(f"Matrices de confusion et rapport sauvegardés dans `{REPORTS_DIR}/`.")


if __name__ == "__main__":
    main()
