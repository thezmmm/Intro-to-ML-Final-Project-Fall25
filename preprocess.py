import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_data(input_file, output_file="./data/processed_data.csv"):
    """
    process raw data and save it as processed data.csv

    process：
    - delete id and date
    - partlybad -> 0/1
    - class4 encode（4 class）
    - Median replaces missing values
    - standardization
    - export
    """

    # read data
    df = pd.read_csv(input_file)

    # delete columns
    drop_cols = [col for col in ["id", "date"] if col in df.columns]
    df = df.drop(columns=drop_cols)

    # partlybad to int
    if "partlybad" in df.columns:
        df["partlybad"] = df["partlybad"].astype(int)

    # Determine if it is a training set
    is_train = "class4" in df.columns
    # class4 label encode
    label_encoder = None
    if is_train:
        label_encoder = LabelEncoder()
        df["class4"] = label_encoder.fit_transform(df["class4"])

        # divide feature and output
        y = df["class4"]
        X = df.drop(columns=["class4"])
    else:
        X = df
        y = None

    # replaces missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    if is_train:
        df_processed["class4"] = y.values

    # save
    df_processed.to_csv(output_file, index=False)

    print(f"process done, saved to {output_file}")

    # return df_processed
    return df_processed,label_encoder

if __name__ == "__main__":
    df_train_processed = preprocess_data("./data/train.csv", "./data/processed_train.csv")
    df_test_processed = preprocess_data("./data/test.csv", "./data/processed_test.csv")


