import pandas as pd
import numpy as np
import joblib
from preprocess import preprocess_data, load_objects, MODEL_DIR

# load models
le, imputer, scaler = load_objects()
clf_4 = joblib.load(f"{MODEL_DIR}/model_4class.pkl")
clf_2 = joblib.load(f"{MODEL_DIR}/model_2class.pkl")

# read test data
df_test = pd.read_csv("./data/test.csv")
ids = df_test["id"].values

# preprocess test data
X_test, _, _, _ = preprocess_data("./data/test.csv")

# 4 class predict
proba_4class = clf_4.predict_proba(X_test)
pred_idx = np.argmax(proba_4class, axis=1)
pred_labels = le.inverse_transform(pred_idx)

# 2 class predict (According to the scoring rules, p is the possibility of the binary classification task)
p_event = clf_2.predict_proba(X_test)[:, 1]

# build submission
submission = pd.DataFrame({
    "id": ids,
    "class4": pred_labels,
    "p": p_event
})

# save
submission.to_csv("./data/submission.csv", index=False)
print("Submission saved to ./data/submission.csv")
