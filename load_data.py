# ============================================================
# load_data.py — Chargement et fusion des datasets
# ============================================================

import pandas as pd


def load_single_file(filepath):
    """
    Charge un fichier dont tous les tweets sont sur UNE SEULE LIGNE
    séparés par des virgules.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig']

    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()

            raw_tweets = content.split(',')
            tweets = [t.strip() for t in raw_tweets if t.strip()]

            print(f"   ✅ {filepath} chargé ({enc}) — {len(tweets)} tweets")
            return pd.DataFrame({"text": tweets})

        except (UnicodeDecodeError, FileNotFoundError) as e:
            if isinstance(e, FileNotFoundError):
                print(f"   ❌ Fichier introuvable : {filepath}")
                return pd.DataFrame(columns=["text"])
            continue

    print(f"   ❌ Impossible de lire : {filepath}")
    return pd.DataFrame(columns=["text"])


def load_data():
    """Charge et fusionne les 3 fichiers avec leurs labels."""
    print("=" * 60)
    print(" Chargement des données...")

    neg = load_single_file("processedNegative.csv")
    neu = load_single_file("processedNeutral.csv")
    pos = load_single_file("processedPositive.csv")

    neg["label"] = "negative"
    neu["label"] = "neutral"
    pos["label"] = "positive"

    df = pd.concat([neg, neu, pos], ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]

    # Supprimer les doublons exacts
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    after = len(df)
    if before != after:
        print(f"   🧹 {before - after} doublons supprimés")

    df = df.reset_index(drop=True)

    print(f"\n Dataset chargé")
    print(f"   Total tweets  : {len(df)}")
    print(f"   Negative      : {(df['label'] == 'negative').sum()}")
    print(f"   Neutral       : {(df['label'] == 'neutral').sum()}")
    print(f"   Positive      : {(df['label'] == 'positive').sum()}")
    print("=" * 60)
    return df