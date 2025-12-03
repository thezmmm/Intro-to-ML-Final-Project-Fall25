import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from preprocess import preprocess_data

def train_four_class(X_train, X_test, y_train, y_test):
    # initialize XGBoost multi-classifier
    clf = LogisticRegression(
        solver='lbfgs',     
        max_iter=500,       
        random_state=42
    )

    # train
    clf.fit(X_train, y_train)

    # predict
    y_pred = clf.predict(X_test)
    # eval
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return clf


def train_two_class(X_train, X_test, y_train, y_test):


    clf = LogisticRegression(
        solver='lbfgs',     
        max_iter=500,       
        random_state=42
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return clf

if __name__ == "__main__":
    df, le = preprocess_data("./data/train.csv")
    # 4 class
    # spilt feature and label
    X = df.drop("class4", axis=1)
    y = df["class4"]
    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42, stratify=y
    )
    clf_4 = train_four_class(X_train, X_test, y_train, y_test)

    # 2 class
    y_orig = le.inverse_transform(df['class4'])
    y = np.array([0 if label == 'nonevent' else 1 for label in y_orig])
    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42, stratify=y
    )
    clf_2 = train_two_class(X_train, X_test, y_train, y_test)
