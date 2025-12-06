import os

import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from preprocess import preprocess_data, save_objects

MODEL_DIR = "./models"

def train_four_class(X_train, X_test, y_train, y_test,sample_weight=None):

    # initialize XGBoost multi-classifier
    clf = xgb.XGBClassifier(
        objective='multi:softprob',  # output percentage
        num_class=len(np.unique(y_train)),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42
    )

    # train
    clf.fit(X_train, y_train,sample_weight=sample_weight)

    # predict
    y_pred = clf.predict(X_test)
    # eval
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return clf


def train_two_class(X_train, X_test, y_train, y_test):

    clf = LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    clf = CalibratedClassifierCV(clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return clf

if __name__ == "__main__":
    df, le, imputer, scaler = preprocess_data("./data/train.csv")
    save_objects(le, imputer, scaler)
    # 4 class
    # spilt feature and label
    X = df.drop("class4", axis=1)
    y = df["class4"]
    y_orig = le.inverse_transform(y)
    weights = {"Ia": 0.057778, "Ib": 0.182222, "II": 0.26, "nonevent": 0.5}
    sample_weight = pd.Series(y_orig).map(weights)
    # split train and test set
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.05, random_state=42, stratify=y
    # )
    clf_4 = train_four_class(X, X, y, y,sample_weight)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf_4, os.path.join(MODEL_DIR, "model_4class.pkl"))

    # 2 class
    y_orig = le.inverse_transform(df['class4'])
    y = np.array([0 if label == 'nonevent' else 1 for label in y_orig])
    # split train and test set
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.05, random_state=42, stratify=y
    # )
    clf_2 = train_two_class(X, X, y, y)
    joblib.dump(clf_2, os.path.join(MODEL_DIR, "model_2class.pkl"))

