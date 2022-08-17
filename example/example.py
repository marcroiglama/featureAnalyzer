import pandas as pd
from feature_analyzer import FeatureAnalyzer

numeric_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
categorical_features = ["sepal_length_cat"]
categorical_labels = ["class"]
numerical_labels = ["petal_width"]

df = pd.read_csv("example/iris.csv")
df["sepal_length_cat"] = df["sepal_length"].round()
analyzer = FeatureAnalyzer(df, numeric_features, numerical_labels, categorical_features, categorical_labels)
report = analyzer.abstract()
print(report)
