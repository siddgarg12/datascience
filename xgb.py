import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Load data (binary classification example)
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Initialize the classifier
    model = xgb.XGBClassifier(
        n_estimators=500,           # number of boosting rounds
        max_depth=6,                # tree depth
        learning_rate=0.05,         # shrinkage / eta
        subsample=0.8,              # row sampling per tree
        colsample_bytree=0.8,       # feature sampling per tree
        reg_alpha=0.0,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",         # fast histogram algorithm
        early_stopping_rounds=20,   # stop if no improvement
        random_state=42,
        n_jobs=-1,
    )

    # 4. Fit with an evaluation set so early stopping kicks in
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    print(f"Best iteration: {model.best_iteration}")

    # 5. Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 6. Evaluate
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, y_proba):.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Cross-validation sanity check (optional)
    cv_scores = cross_val_score(
        xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        eval_metric="logloss", tree_method="hist", random_state=42),
        X, y, cv=5, scoring="roc_auc"
    )
    print(f"\n5-fold CV ROC AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # 8. Feature importance
    importances = model.feature_importances_
    order = np.argsort(importances)[-10:]  # top 10
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(order)), importances[order])
    plt.yticks(range(len(order)), [feature_names[i] for i in order])
    plt.xlabel("Importance (gain)")
    plt.title("Top 10 features")
    plt.tight_layout()
    plt.savefig("xgb_feature_importance.png", dpi=120)

if __name__ == "__main__":
    main()