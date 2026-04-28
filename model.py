"""
Loan default risk prediction model.
Builds and evaluates classification models with AUC, KS statistic, and Gini coefficient.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        roc_auc_score, roc_curve, precision_recall_curve
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class LoanRiskModel:
    """
    Binary classification pipeline for loan default prediction.
    Computes AUC-ROC, KS statistic, Gini coefficient, and scorecard bands.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 target_col: str = "default"):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.models: Dict[str, Pipeline] = {}
        self.results: List[Dict] = []
        self.best_model_name: Optional[str] = None
        self._best_pipe: Optional[Pipeline] = None

    def _preprocessor(self):
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat",
                                  OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                  self.categorical_features))
        from sklearn.compose import ColumnTransformer
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _estimators(self) -> Dict:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=500, C=1.0),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42,
                                                    class_weight="balanced", n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                                             max_depth=4, random_state=42),
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05,
                                                   max_depth=5, scale_pos_weight=3,
                                                   random_state=42, verbosity=0,
                                                   tree_method="hist", eval_metric="auc")
        return models

    def _ks_statistic(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Kolmogorov-Smirnov statistic: max difference between default and non-default CDFs."""
        df = pd.DataFrame({"y": y_true, "prob": y_prob}).sort_values("prob", ascending=False)
        n_pos = (y_true == 1).sum()
        n_neg = (y_true == 0).sum()
        df["cum_pos"] = (df["y"] == 1).cumsum() / n_pos
        df["cum_neg"] = (df["y"] == 0).cumsum() / n_neg
        return float((df["cum_pos"] - df["cum_neg"]).abs().max())

    def _gini(self, auc: float) -> float:
        return 2 * auc - 1

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        feat_cols = self.numeric_features + self.categorical_features
        df = df[feat_cols + [self.target_col]].dropna(subset=[self.target_col])
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        X = df[feat_cols]
        y = df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        preprocessor = self._preprocessor()
        self.results = []

        for name, est in self._estimators().items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", est)])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            y_pred = pipe.predict(X_test)
            auc = float(roc_auc_score(y_test, y_prob))
            ks = self._ks_statistic(y_test.values, y_prob)
            gini = self._gini(auc)
            acc = float(accuracy_score(y_test, y_pred))
            self.models[name] = pipe
            self.results.append({
                "model": name,
                "auc": round(auc, 4),
                "ks": round(ks, 4),
                "gini": round(gini, 4),
                "accuracy": round(acc, 4),
            })

        results_df = pd.DataFrame(self.results).sort_values("auc", ascending=False).reset_index(drop=True)
        self.best_model_name = results_df.iloc[0]["model"]
        self._best_pipe = self.models[self.best_model_name]
        return results_df

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self._best_pipe is None:
            raise RuntimeError("Call fit() first.")
        feat_cols = self.numeric_features + self.categorical_features
        return self._best_pipe.predict_proba(df[feat_cols])[:, 1]

    def risk_band(self, probabilities: np.ndarray) -> pd.Series:
        """Assign risk bands: low, medium, high, very_high."""
        bands = pd.cut(
            probabilities,
            bins=[0, 0.15, 0.35, 0.60, 1.0],
            labels=["low", "medium", "high", "very_high"],
            include_lowest=True,
        )
        return bands

    def scorecard(self, probabilities: np.ndarray,
                  min_score: int = 300, max_score: int = 850) -> np.ndarray:
        """
        Map default probabilities to a credit score scale (higher = less risky).
        Uses a log-odds linear scaling.
        """
        eps = 1e-7
        p = np.clip(probabilities, eps, 1 - eps)
        log_odds = np.log(p / (1 - p))
        score_range = max_score - min_score
        mid_score = (min_score + max_score) / 2
        scores = mid_score - (log_odds * score_range / 6)
        return np.clip(scores, min_score, max_score).astype(int)

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self.best_model_name not in self.models:
            return None
        pipe = self.models[self.best_model_name]
        est = pipe.named_steps["model"]
        if not hasattr(est, "feature_importances_"):
            return None
        prep = pipe.named_steps["preprocessor"]
        try:
            cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(self.categorical_features))
        except Exception:
            cat_names = []
        names = self.numeric_features + cat_names
        return pd.DataFrame({
            "feature": names[:len(est.feature_importances_)],
            "importance": est.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    np.random.seed(42)
    n = 5000
    age = np.random.randint(21, 70, n)
    income = np.random.lognormal(10.5, 0.6, n)
    loan_amount = np.random.lognormal(11, 0.7, n)
    loan_term = np.random.choice([12, 24, 36, 48, 60], n)
    credit_score = np.random.randint(300, 850, n)
    delinquencies = np.random.poisson(0.3, n)
    employment_type = np.random.choice(["salaried", "self_employed", "business"], n)
    loan_purpose = np.random.choice(["home", "education", "business", "personal", "vehicle"], n)
    p_default = 1 / (1 + np.exp(
        0.01 * credit_score - 0.00001 * income - 0.01 * age + 0.5 * delinquencies - 5
    ))
    default = (np.random.rand(n) < p_default).astype(int)

    df = pd.DataFrame({
        "age": age.astype(float), "income": income,
        "loan_amount": loan_amount, "loan_term": loan_term.astype(float),
        "credit_score": credit_score.astype(float),
        "delinquencies": delinquencies.astype(float),
        "employment_type": employment_type, "loan_purpose": loan_purpose,
        "default": default,
    })
    print(f"Default rate: {default.mean():.2%}")

    model = LoanRiskModel(
        numeric_features=["age", "income", "loan_amount", "loan_term", "credit_score", "delinquencies"],
        categorical_features=["employment_type", "loan_purpose"],
    )
    results = model.fit(df)
    print("\nModel comparison:")
    print(results.to_string(index=False))

    sample_probs = model.predict_proba(df.head(10))
    scores = model.scorecard(sample_probs)
    bands = model.risk_band(sample_probs)
    for prob, score, band in zip(sample_probs[:5], scores[:5], bands[:5]):
        print(f"  P(default)={prob:.3f} | Score={score} | Band={band}")

    imp = model.feature_importance()
    if imp is not None:
        print("\nTop features:")
        print(imp.head(5).to_string(index=False))
