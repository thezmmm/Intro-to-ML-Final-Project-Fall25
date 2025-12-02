import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder

path = './data/'

df = pd.read_csv(path+"train.csv")

df.info()
print(df.describe().T)
print("=== check missing value ===")
print(df.isna().sum().sort_values(ascending=False))
print("=== check duplicate value ===")
print(df.duplicated().sum())

# check the distribution of class4
df['class4'].value_counts(normalize=True)
df['class4'].value_counts().plot(kind='bar')

# hist for all param
df.hist(figsize=(20, 25), bins=40)

mean_cols = [c for c in df.columns if c.endswith(".mean")]
std_cols = [c for c in df.columns if c.endswith(".std")]
print("=== check mean value ===")
print(df[mean_cols].describe().T)
print("=== check std value ===")
print(df[std_cols].describe().T)

# Correlation Heatmap of MEAN features
plt.figure(figsize=(18, 16))
sns.heatmap(df[mean_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of MEAN features")

# Relationship between features and class4
X = df[mean_cols + std_cols]
y = df['class4']

f_vals, p_vals = f_classif(X, y)

importance = pd.DataFrame({
    'feature': X.columns,
    'f_value': f_vals,
    'p_value': p_vals
}).sort_values("f_value", ascending=False)

print("Top 20 with highest ANOVA F-score")
print(importance.head(20))

# distribution differences of the top 5 features across different Class 4
top_features = importance.feature.head(5)
for f in top_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='class4', y=f)
    plt.title(f"{f} by class4")
    plt.show()

# find the most important features by random forest
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
print("=== best features by random forest ===")
print(importances.sort_values(ascending=False).head(20))


# pca
X = df[mean_cols + std_cols]
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
y = df['class4']
le = LabelEncoder()
y_num = le.fit_transform(y)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y_num,cmap='tab10')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.title("PCA Projection colored by class4")
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())


