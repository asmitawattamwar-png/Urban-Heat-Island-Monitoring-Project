"""
Member 2 / Part 6: PCA on the supplied cleaned UHI dataset.

PCA is applied to standardized numeric environmental/socioeconomic predictors.
Temperature is excluded from PCA predictors so the components describe the input
features rather than being dominated by the outcome that regression predicts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "cleaned_uhi_dataset.csv"
OUT = ROOT / "outputs" / "member2" / "pca"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "Temperature (°C)"
EXCLUDE = {"City Name", "Land Cover", TARGET, "Latitude", "Longitude"}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT}\nRun data_preprocessing.py first.")
    df = pd.read_csv(INPUT)
    numeric = df.select_dtypes(include=np.number)
    features = [c for c in numeric.columns if c not in EXCLUDE]
    if len(features) < 2:
        raise ValueError(f"Need at least 2 numeric predictor columns; found {features}")
    X = numeric[features].replace([np.inf, -np.inf], np.nan)
    # Use complete rows for a transparent PCA fit; cleaned file is expected to be complete.
    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    if len(X) < 3:
        raise ValueError("Too few complete observations for PCA.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    variance = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(features))],
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_)
    })
    variance.to_csv(OUT / "pca_explained_variance.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(len(features))]
    )
    loadings.index.name = "feature"
    loadings.to_csv(OUT / "pca_loadings.csv")

    score_df = pd.DataFrame(scores[:, :min(3, scores.shape[1])],
                            columns=[f"PC{i+1}_score" for i in range(min(3, scores.shape[1]))],
                            index=X.index)
    score_df.insert(0, "Land Cover", df.loc[X.index, "Land Cover"].astype(str).values if "Land Cover" in df.columns else "Unknown")
    score_df.to_csv(OUT / "pca_scores_first_components.csv", index_label="row_index")

    # Scree / cumulative variance plot.
    x = np.arange(1, len(features) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x, pca.explained_variance_ratio_, alpha=0.8, label="Individual explained variance")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in x])
    ax2 = ax1.twinx()
    ax2.plot(x, np.cumsum(pca.explained_variance_ratio_), marker="o", label="Cumulative explained variance")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.05)
    plt.title("PCA Scree Plot and Cumulative Explained Variance")
    fig.tight_layout()
    fig.savefig(OUT / "pca_scree_plot.png", dpi=180)
    plt.close(fig)

    # PC1 vs PC2 colored by land-cover label (category is used only for display).
    if scores.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        labels = df.loc[X.index, "Land Cover"].astype(str) if "Land Cover" in df.columns else pd.Series("All", index=X.index)
        for label in sorted(labels.unique()):
            mask = labels.to_numpy() == label
            plt.scatter(scores[mask, 0], scores[mask, 1], alpha=0.65, s=22, label=label)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
        plt.title("PCA Score Plot (PC1 vs PC2)")
        plt.legend(title="Land Cover", fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT / "pca_pc1_pc2.png", dpi=180)
        plt.close()

    # Show strongest absolute loadings for PC1 and PC2.
    top_rows = []
    for pc in [c for c in ["PC1", "PC2", "PC3"] if c in loadings.columns]:
        for feature, value in loadings[pc].abs().sort_values(ascending=False).head(5).items():
            top_rows.append({"component": pc, "feature": feature, "loading": loadings.loc[feature, pc],
                             "absolute_loading": abs(loadings.loc[feature, pc])})
    pd.DataFrame(top_rows).to_csv(OUT / "top_pca_loadings.csv", index=False)

    note = (
        f"Input rows used: {len(X)}\nFeatures used ({len(features)}): {', '.join(features)}\n"
        "Features excluded: City Name, Land Cover, Temperature (°C), Latitude, Longitude (if present).\n"
        "Numeric predictors were standardized using StandardScaler before PCA.\n"
        "PCA components are linear combinations of the standardized predictors. Loading signs can flip without changing the PCA solution.\n"
        "This tabular dataset has no raster grid or pixel coordinates tied to spectral measurements, so a true spatial PC-score map cannot be produced from these columns alone.\n"
    )
    (OUT / "pca_notes.txt").write_text(note)
    print(f"Saved PCA outputs to: {OUT}")
    print(variance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
