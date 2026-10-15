"""
Member 2 / Part 7: SVD and low-rank reconstruction.

Uses the same standardized numeric predictor matrix as pca_analysis.py, making
the PCA/SVD relationship directly comparable. For centered/scaled X:
X = U Sigma V^T; PCA directions are the right singular vectors, and PCA
explained-variance ratios equal Sigma_i^2 / sum(Sigma_j^2).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "cleaned_uhi_dataset.csv"
OUT = ROOT / "outputs" / "member2" / "svd"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "Temperature (°C)"
EXCLUDE = {"City Name", "Land Cover", TARGET, "Latitude", "Longitude"}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {INPUT}\nRun data_preprocessing.py first.")
    df = pd.read_csv(INPUT)
    numeric = df.select_dtypes(include=np.number)
    features = [c for c in numeric.columns if c not in EXCLUDE]
    Xdf = numeric[features].replace([np.inf, -np.inf], np.nan)
    Xdf = Xdf.loc[Xdf.notna().all(axis=1)]
    if len(features) < 2 or len(Xdf) < 3:
        raise ValueError("SVD requires at least 2 numeric features and 3 complete rows.")

    # StandardScaler centers and scales columns; SVD is applied to this matrix.
    scaler = StandardScaler()
    X = scaler.fit_transform(Xdf)
    U, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
    energy = singular_values**2
    ratios = energy / energy.sum()
    cumulative = np.cumsum(ratios)
    svd_summary = pd.DataFrame({
        "component": [f"SV{i+1}" for i in range(len(singular_values))],
        "singular_value": singular_values,
        "squared_singular_value_energy": energy,
        "explained_energy_ratio": ratios,
        "cumulative_explained_energy_ratio": cumulative
    })
    svd_summary.to_csv(OUT / "svd_singular_values_and_energy.csv", index=False)

    # Right singular vectors / feature weights.
    pd.DataFrame(Vt.T, index=features,
                 columns=[f"SV{i+1}" for i in range(len(singular_values))]
                 ).rename_axis("feature").to_csv(OUT / "svd_right_singular_vectors.csv")
    # U contains left singular vectors (observation directions).
    pd.DataFrame(U[:, :min(3, U.shape[1])],
                 columns=[f"U_component_{i+1}" for i in range(min(3, U.shape[1]))],
                 index=Xdf.index).to_csv(OUT / "svd_left_singular_vectors_first3.csv", index_label="row_index")

    rows = []
    reconstructions = {}
    for k in sorted(set([1, 2, 3, min(5, len(singular_values)), min(10, len(singular_values)), len(singular_values)])):
        if k < 1:
            continue
        Xk = (U[:, :k] * singular_values[:k]) @ Vt[:k, :]
        residual = X - Xk
        rows.append({
            "components_k": k,
            "retained_energy_ratio": energy[:k].sum() / energy.sum(),
            "reconstruction_rmse_standardized": float(np.sqrt(np.mean(residual**2))),
            "reconstruction_mae_standardized": float(np.mean(np.abs(residual))),
            "frobenius_reconstruction_error": float(np.linalg.norm(residual, ord="fro"))
        })
        reconstructions[k] = Xk
    reconstruction = pd.DataFrame(rows)
    reconstruction.to_csv(OUT / "svd_reconstruction_metrics.csv", index=False)

    # Plot singular-value energy and reconstruction RMSE.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(ratios)+1), cumulative, marker="o")
    ax.set_xlabel("Number of retained components")
    ax.set_ylabel("Cumulative retained squared-singular-value energy")
    ax.set_ylim(0, 1.05)
    ax.set_title("SVD Cumulative Retained Energy")
    fig.tight_layout()
    fig.savefig(OUT / "svd_cumulative_energy.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    plt.plot(reconstruction["components_k"], reconstruction["reconstruction_rmse_standardized"], marker="o")
    plt.xlabel("Number of retained components (k)")
    plt.ylabel("Reconstruction RMSE (standardized scale)")
    plt.title("SVD Low-Rank Reconstruction Error")
    plt.tight_layout()
    plt.savefig(OUT / "svd_reconstruction_error.png", dpi=180)
    plt.close()

    # Reconstruct with the smallest k reaching at least 90% of squared-singular-value energy.
    k90 = int(np.searchsorted(cumulative, 0.90) + 1)
    X90 = (U[:, :k90] * singular_values[:k90]) @ Vt[:k90, :]
    reconstructed_original_scale = scaler.inverse_transform(X90)
    recon_df = pd.DataFrame(reconstructed_original_scale, columns=features, index=Xdf.index)
    recon_df.to_csv(OUT / "svd_reconstruction_at_90pct_energy.csv", index_label="row_index")

    (OUT / "svd_notes.txt").write_text(
        f"Input rows used: {len(Xdf)}\nFeatures used ({len(features)}): {', '.join(features)}\n"
        f"Components required to retain >=90% squared-singular-value energy: {k90}\n"
        f"Energy retained at k={k90}: {cumulative[k90-1]:.6f}\n"
        "The reconstruction RMSE and MAE are reported on the standardized scale, so errors are comparable across differently scaled variables.\n"
        "PCA and SVD are related: for the standardized matrix X, right singular vectors are PCA directions, and squared singular values normalized by their sum give the PCA explained-variance ratios (up to numerical precision and the variance denominator convention).\n"
        "The saved 90%-energy reconstruction is inverse-transformed to the original units for readability; it is an approximation, not cleaned source data.\n"
    )
    print(f"Saved SVD outputs to: {OUT}")
    print(svd_summary.head(10).to_string(index=False))
    print("\nReconstruction metrics:")
    print(reconstruction.to_string(index=False))


if __name__ == "__main__":
    main()
