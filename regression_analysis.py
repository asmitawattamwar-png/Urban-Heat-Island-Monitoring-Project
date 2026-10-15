"""Member 2 / Part 9: defensible regression analysis for the supplied cleaned UHI CSV.

The script compares a mean baseline with several models using 5-fold CV on the
training partition, chooses the lowest-CV-RMSE candidate, and evaluates it once
on a held-out test partition. It never manufactures target relationships or
reports a weak model as strong. This dataset has very weak predictors of temp.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'outputs' / 'cleaned_uhi_dataset.csv'
OUT = ROOT / 'outputs' / 'member2' / 'regression'
OUT.mkdir(parents=True, exist_ok=True)
TARGET = 'Temperature (°C)'
SEED = 42


def adjusted_r2(r2, n, p):
    return np.nan if n <= p + 1 else 1 - (1-r2)*(n-1)/(n-p-1)


def build_pipeline(X, estimator, scale_numeric=True):
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_steps.append(('scaler', StandardScaler()))
    transformers = []
    if num:
        transformers.append(('numeric', Pipeline(num_steps), num))
    if cat:
        transformers.append(('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat))
    return Pipeline([('preprocess', ColumnTransformer(transformers, remainder='drop')),
                     ('model', estimator)])


def metric_row(name, y_true, y_pred, n_features, n_train, n_test):
    r2 = r2_score(y_true, y_pred)
    return {'model': name, 'n_train': n_train, 'n_test': n_test,
            'predictor_count': n_features, 'R2_test': r2,
            'adjusted_R2_test': adjusted_r2(r2, len(y_true), n_features),
            'MAE_test_C': mean_absolute_error(y_true, y_pred),
            'RMSE_test_C': float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'Cleaned dataset not found: {INPUT}. Put the CSV at outputs/cleaned_uhi_dataset.csv.')
    df = pd.read_csv(INPUT)
    if TARGET not in df:
        raise ValueError(f'Missing target column: {TARGET}')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET]).copy()
    y = pd.to_numeric(df[TARGET], errors='coerce')
    df = df.loc[y.notna()].copy(); y = y.loc[y.notna()]
    # Exclude record IDs and raw coordinates: no documented spatial sampling design.
    all_cols = [c for c in df.columns if c not in {TARGET, 'City Name', 'Latitude', 'Longitude'}]
    # Candidate predictor sets are chosen before evaluation, with a simple land-cover
    # baseline and environmental models. Land Cover is a legitimate categorical feature.
    candidate_sets = {
        'Land-cover-only Ridge': ['Land Cover'] if 'Land Cover' in df.columns else [],
        'All available environmental predictors - Ridge': all_cols,
        'All available environmental predictors - Random Forest': all_cols,
        'All available environmental predictors - Extra Trees': all_cols,
        'All available environmental predictors - Gradient Boosting': all_cols,
    }
    candidate_sets = {k:v for k,v in candidate_sets.items() if v}
    X_all = {k: df[v].copy() for k,v in candidate_sets.items()}
    X_train_idx, X_test_idx = train_test_split(df.index, test_size=.20, random_state=SEED)
    y_train, y_test = y.loc[X_train_idx], y.loc[X_test_idx]
    estimators = {
        'Land-cover-only Ridge': Ridge(alpha=10.0),
        'All available environmental predictors - Ridge': Ridge(alpha=10.0),
        'All available environmental predictors - Random Forest': RandomForestRegressor(n_estimators=250, min_samples_leaf=8, max_features=.8, random_state=SEED, n_jobs=-1),
        'All available environmental predictors - Extra Trees': ExtraTreesRegressor(n_estimators=250, min_samples_leaf=8, max_features=.8, random_state=SEED, n_jobs=-1),
        'All available environmental predictors - Gradient Boosting': GradientBoostingRegressor(n_estimators=80, learning_rate=.03, max_depth=2, loss='huber', random_state=SEED),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    pipelines = {}
    for name, X in X_all.items():
        est = estimators[name]
        pipe = build_pipeline(X.loc[X_train_idx], est, scale_numeric=True)
        pipelines[name] = pipe
        scores = cross_validate(pipe, X.loc[X_train_idx], y_train, cv=cv,
            scoring={'r2':'r2', 'mae':'neg_mean_absolute_error', 'rmse':'neg_root_mean_squared_error'},
            n_jobs=1, return_train_score=False)
        cv_rows.append({'model':name, 'cv_R2_mean':scores['test_r2'].mean(),
            'cv_R2_std':scores['test_r2'].std(ddof=1),
            'cv_MAE_C_mean':-scores['test_mae'].mean(),
            'cv_RMSE_C_mean':-scores['test_rmse'].mean(),
            'predictors':', '.join(X.columns)})
    cv_table = pd.DataFrame(cv_rows).sort_values('cv_RMSE_C_mean').reset_index(drop=True)
    cv_table.to_csv(OUT/'cross_validation_model_comparison.csv', index=False)
    best_name = cv_table.iloc[0]['model']
    best_X = X_all[best_name]
    best_pipe = pipelines[best_name]
    best_pipe.fit(best_X.loc[X_train_idx], y_train)
    best_pred = best_pipe.predict(best_X.loc[X_test_idx])

    # Mean baseline is an essential benchmark, not a candidate to hide.
    baseline = DummyRegressor(strategy='mean').fit(np.zeros((len(y_train),1)), y_train)
    baseline_pred = baseline.predict(np.zeros((len(y_test),1)))
    baseline_row = metric_row('Training-mean baseline', y_test, baseline_pred, 0, len(y_train), len(y_test))
    best_row = metric_row(best_name, y_test, best_pred, best_X.shape[1], len(y_train), len(y_test))
    metrics = pd.DataFrame([baseline_row, best_row])
    metrics.to_csv(OUT/'model_evaluation_metrics.csv', index=False)

    pred_df = pd.DataFrame({'actual_temperature_C': y_test,
                            'training_mean_baseline_C': baseline_pred,
                            'selected_model_prediction_C': best_pred,
                            'residual_C': y_test.to_numpy()-best_pred}, index=y_test.index)
    pred_df.index.name = 'row_index'; pred_df.to_csv(OUT/'test_set_predictions.csv')

    # Actual vs predicted plot.
    lo = float(min(y_test.min(), np.min(best_pred), np.min(baseline_pred)))
    hi = float(max(y_test.max(), np.max(best_pred), np.max(baseline_pred)))
    plt.figure(figsize=(7,6)); plt.scatter(y_test, best_pred, alpha=.75, label=best_name)
    plt.scatter(y_test, baseline_pred, alpha=.45, label='Training-mean baseline')
    plt.plot([lo,hi],[lo,hi],'--',label='Ideal prediction')
    plt.xlabel('Actual temperature (°C)'); plt.ylabel('Predicted temperature (°C)')
    plt.title('Held-out test set: actual vs predicted temperature'); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUT/'actual_vs_predicted.png', dpi=180); plt.close()

    resid = y_test.to_numpy()-best_pred
    plt.figure(figsize=(7,5)); plt.scatter(best_pred,resid,alpha=.75); plt.axhline(0,ls='--')
    plt.xlabel('Predicted temperature (°C)'); plt.ylabel('Residual (actual − predicted, °C)')
    plt.title('Selected model residual plot'); plt.tight_layout(); plt.savefig(OUT/'selected_model_residuals.png',dpi=180); plt.close()
    plt.figure(figsize=(7,5)); plt.hist(resid,bins=20,edgecolor='black')
    plt.xlabel('Residual (°C)'); plt.ylabel('Count'); plt.title('Selected model residual distribution')
    plt.tight_layout(); plt.savefig(OUT/'selected_model_residual_histogram.png',dpi=180); plt.close()

    notes = [
        f'Target: {TARGET}', f'Rows used: {len(df)}',
        'Evaluation: fixed 80/20 holdout (random_state=42); model selection used 5-fold CV on training data only.',
        f'Selected by lowest mean CV RMSE: {best_name}',
        f'Mean CV RMSE (selected model): {cv_table.iloc[0]["cv_RMSE_C_mean"]:.4f} °C',
        f'Test R2 (selected model): {best_row["R2_test"]:.4f}',
        f'Test MAE (selected model): {best_row["MAE_test_C"]:.4f} °C',
        f'Test RMSE (selected model): {best_row["RMSE_test_C"]:.4f} °C',
        f'Test R2 (training-mean baseline): {baseline_row["R2_test"]:.4f}',
        '', 'INTERPRETATION / LIMITATION:',
        'The target has near-zero linear correlations with the available numerical predictors. The data also has no documented date/season, satellite bands, NDVI, NDBI, or albedo.',
        'If the selected model does not outperform the mean baseline, this is evidence that the supplied dataset does not support reliable temperature prediction. Do not manipulate the target, leak it into features, or report a fabricated high R2.',
        'The source records appear synthetic or at least lack sufficient provenance for a satellite-based UHI claim. Treat all results as exploratory until the dataset source and measurement definitions are verified.',
        'Adjusted R2 on a held-out test set is included as a descriptive statistic only; cross-validation and holdout RMSE are more useful for model comparison.'
    ]
    (OUT/'regression_notes.txt').write_text('\n'.join(notes)+'\n', encoding='utf-8')
    print('Cross-validation comparison:')
    print(cv_table[['model','cv_R2_mean','cv_MAE_C_mean','cv_RMSE_C_mean']].to_string(index=False))
    print('\nSelected model:', best_name)
    print('\nHoldout metrics:'); print(metrics.to_string(index=False))
    print('\nOutputs:', OUT)

if __name__ == '__main__':
    main()
