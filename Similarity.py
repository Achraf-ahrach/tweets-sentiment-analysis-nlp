# ============================================================
# similarity.py — Similarité cosinus, Top-10 paires similaires
# ============================================================

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def top10_similar_pairs(X_matrix, df_texts, method_name, vec_name,
                        sample_size=1000):
    """
    Trouve les 10 paires de tweets les plus similaires.

    Paramètres :
        X_matrix    : matrice vectorisée des tweets (TF-IDF)
        df_texts    : Series pandas avec les textes preprocessés
        method_name : nom de la méthode de preprocessing (pour affichage)
        vec_name    : nom du vectoriseur (pour affichage)
        sample_size : nombre de tweets analysés (limité pour la performance)
    """
    print(f"\nTop-10 paires similaires — [{method_name}] + [{vec_name}]")
    print("-" * 60)

    # Limiter à sample_size tweets pour éviter des calculs trop lourds
    n = min(sample_size, X_matrix.shape[0])
    X_sample      = X_matrix[:n]
    texts_sample  = df_texts.iloc[:n].reset_index(drop=True)

    # Calcul de la matrice de similarité cosinus
    sim_matrix = cosine_similarity(X_sample)

    # Ignorer la diagonale (un tweet est toujours identique à lui-même)
    np.fill_diagonal(sim_matrix, 0)

    # Ne garder que le triangle supérieur pour éviter les doublons (i,j) et (j,i)
    upper_tri  = np.triu(sim_matrix, k=1)
    rows_all, cols_all = np.where(upper_tri > 0)
    scores_all = upper_tri[rows_all, cols_all]

    # Trier par score décroissant
    sorted_idx = np.argsort(scores_all)[::-1]

    printed = 0
    seen    = set()

    for idx in sorted_idx:
        if printed >= 10:
            break

        i, j  = rows_all[idx], cols_all[idx]
        score = scores_all[idx]
        ta    = texts_sample.iloc[i]
        tb    = texts_sample.iloc[j]

        # Ignorer les paires dont les textes sont identiques
        if ta == tb:
            continue

        # Ignorer les paires déjà affichées
        pair_key = tuple(sorted([ta, tb]))
        if pair_key in seen:
            continue

        seen.add(pair_key)
        printed += 1
        print(f"#{printed:2d} Score={score:.4f}")
        print(f"     A: {ta[:90]}")
        print(f"     B: {tb[:90]}")