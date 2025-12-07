import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import numpy as np

def objective(trial, X, y):
    # Optuna自动选取的参数
    param = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42
    }

    # 使用 5-fold CV
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logloss_scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_val)
        logloss = log_loss(y_val, y_pred)
        logloss_scores.append(logloss)

    return np.mean(logloss_scores)

import pandas as pd
from preprocess import preprocess_data

# 预处理
df, le, imputer, scaler = preprocess_data("./data/train.csv")
X = df.drop("class4", axis=1).values
y = df["class4"].values

# 创建 study
study = optuna.create_study(direction='minimize')  # logloss 越小越好
study.optimize(lambda trial: objective(trial, X, y), n_trials=100)  # 50次尝试

# 输出最优参数
print("Best trial:")
trial = study.best_trial
print(trial.params)


