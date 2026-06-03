# ============================================================
# main.py — Fichier principal, relie tous les modules
# ============================================================
#
# Structure du projet :
#   main.py           ← Lance tout (ce fichier)
#   load_data.py      ← Chargement des données
#   preprocessing.py  ← 6 méthodes de preprocessing
#   vectorization.py  ← 3 vectoriseurs (binary, count, tfidf)
#   similarity.py     ← Top-10 paires similaires (cosinus)
#   models.py         ← 4 algorithmes ML + évaluation
#   data/
#     processedNegative.csv
#     processedNeutral.csv
#     processedPositive.csv
# ============================================================

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

# ── Import des modules locaux ────────────────────────────────
from load_data      import load_data
from preprocessing  import PREPROCESSING_METHODS
from Vectorization  import get_vectorizers, get_word2vec_vector
from Similarity     import top10_similar_pairs
from Models         import train_and_evaluate, detailed_report


def main():

    # ── 1. Chargement des données ────────────────────────────
    df = load_data()

    all_results  = {}
    best_overall = {
        "acc":     0,
        "model_name":   "",
        "model": None,
        "config":  "",
        "X_train": None,
        "X_test":  None,
        "y_train": None,
        "y_test":  None,
    }

    # ── 2. Boucle 1 : méthodes de preprocessing ─────────────
    for prep_name, prep_func in PREPROCESSING_METHODS.items():

        print(f"\n\n{'#'*60}")
        print(f"# PREPROCESSING : {prep_name.upper()}")
        print(f"{'#'*60}")

        # Appliquer le preprocessing sur tous les tweets
        df_prep = df.copy()
        df_prep["text_clean"] = df_prep["text"].apply(prep_func)

        # Supprimer les tweets devenus vides après preprocessing
        df_prep = df_prep[df_prep["text_clean"].str.strip() != ""]

        X_text = df_prep["text_clean"]
        y      = df_prep["label"]

        # ── 3. Boucle 2 : vectoriseurs ───────────────────────
        for vec_name, vectorizer in get_vectorizers().items():

            config_name = f"{prep_name} + {vec_name}"

            try:
                # Vectorisation
                X = vectorizer.fit_transform(X_text)

                # Split stratifié 80% train / 20% test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )

                # Similarité cosinus top-10 (uniquement avec tfidf)
                if vec_name == "tfidf":
                    top10_similar_pairs(X, X_text, prep_name, vec_name)

                # Entraînement et évaluation des 4 modèles ML
                results = train_and_evaluate(
                    X_train, X_test, y_train, y_test, config_name
                )
                all_results[config_name] = results

                # Suivre la meilleure configuration globale
                best_acc_config = max(r["acc"] for r in results.values()) if results else 0
                if best_acc_config > best_overall["acc"]:
                    best_model_name = max(results, key=lambda k: results[k]["acc"])
                    best_overall.update({
                        "acc":     results[best_model_name]["acc"],
                        "config":  config_name,
                        "model_name": best_model_name,
                        "model":   results[best_model_name]["model"],
                        "X_train": X_train,
                        "X_test":  X_test,
                        "y_train": y_train,
                        "y_test":  y_test,
                    })

            except Exception as e:
                print(f"    Erreur pour {config_name} : {e}")
    # ── 3. Boucle Word2Vec (séparée) ─────────────────────────
    print(f"\n\n{'#'*60}")
    print(f"# BONUS — WORD2VEC")
    print(f"{'#'*60}")

    for prep_name, prep_func in PREPROCESSING_METHODS.items():

        config_name = f"{prep_name} + word2vec"

        try:
            df_prep = df.copy()
            df_prep["text_clean"] = df_prep["text"].apply(prep_func)
            df_prep = df_prep[df_prep["text_clean"].str.strip() != ""]

            X_text = df_prep["text_clean"]
            y      = df_prep["label"]

            # ← Word2Vec directement sans fit_transform
            X = get_word2vec_vector(X_text)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            results = train_and_evaluate(
                X_train, X_test, y_train, y_test, config_name
            )
            all_results[config_name] = results

            best_acc_config = max(r["acc"] for r in results.values()) if results else 0
            if best_acc_config > best_overall["acc"]:
                best_model_name = max(results, key=lambda k: results[k]["acc"])
                best_overall.update({
                    "acc":        results[best_model_name]["acc"],
                    "config":     config_name,
                    "model_name": best_model_name,
                    "model":      results[best_model_name]["model"],
                    "X_train":    X_train,
                    "X_test":     X_test,
                    "y_train":    y_train,
                    "y_test":     y_test,
                })

        except Exception as e:
            print(f"    Erreur pour {config_name} : {e}")
    # ── 4. Résumé final ──────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(" RÉSUMÉ FINAL — TOUTES LES CONFIGURATIONS")
    print(f"{'='*60}")
    print(f"{'Configuration':<40} {'LR':>7} {'NB':>7} {'SVC':>7} {'RF':>7}")
    print("-" * 68)

    for config, res in all_results.items():
        lr  = res.get("Logistic Regression", {}).get("acc", 0)
        nb  = res.get("Naive Bayes",         {}).get("acc", 0)
        svc = res.get("Linear SVC",          {}).get("acc", 0)
        rf  = res.get("Random Forest",       {}).get("acc", 0)
        flag = " Objectif principal atteint" if max(lr, nb, svc, rf) >= 0.832 else ""
        print(f"{config:<40} {lr:>7.4f} {nb:>7.4f} {svc:>7.4f} {rf:>7.4f}{flag}")

    print(f"\n Meilleure configuration : {best_overall['config']}")
    print(f"   Accuracy : {best_overall['acc']:.4f}")

    # ── 5. Rapport détaillé du meilleur modèle ───────────────
    if best_overall["X_train"] is not None:
        detailed_report(
            best_overall["X_train"], best_overall["X_test"],
            best_overall["y_train"], best_overall["y_test"],
            best_overall["config"], best_overall["model"]
        )

    print("\n Analyse complète terminée !")


if __name__ == "__main__":
    main()