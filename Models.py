# ============================================================
# models.py — Entraînement et évaluation des modèles ML
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def get_models():
    """
    Retourne les 4 algorithmes ML à comparer :
      - Logistic Regression : modèle linéaire simple et efficace
      - Naive Bayes         : rapide, fonctionne bien sur du texte
      - Linear SVC          : souvent le meilleur sur du texte
      - Random Forest       : ensemble d'arbres de décision
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs'
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVC":  LinearSVC(max_iter=2000, C=1.0),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, config_name):
    """
    Entraîne et évalue tous les modèles sur une configuration donnée.

    Retourne :
        dict — {nom_modele: accuracy}
    """
    models  = get_models()
    results = {}

    print(f"\n{'='*60}")
    print(f"Configuration : {config_name}")
    print(f"{'='*60}")

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            results[name] = {"acc": acc, "model": model}

            flag = "Objectif principal atteint" if acc >= 0.832 else "  "
            print(f"{flag} {name:<25} Accuracy: {acc:.4f}")

        except Exception as e:
            print(f"{name} erreur : {e}")

    if results:
        best_name = max(results, key=lambda k: results[k]["acc"])
        best_acc  = results[best_name]["acc"]         
        print(f"\nMeilleur : {best_name} → {best_acc:.4f}")

        if best_acc >= 0.873:
            print("Bonus 2 atteint (≥ 0.873) !")
        elif best_acc >= 0.851:
            print("Bonus 1 atteint (≥ 0.851) !")
        elif best_acc >= 0.832:
            print("Objectif principal atteint (≥ 0.832) !")
        else:
            print("Objectif non atteint (< 0.832)")

    return results


def detailed_report(X_train, X_test, y_train, y_test, config_name, model):
    print(f"\n Rapport détaillé — {config_name}")
    print("-" * 60)
    y_pred = model.predict(X_test)

    print(classification_report(
        y_test, y_pred,
        target_names=["negative", "neutral", "positive"]
    ))