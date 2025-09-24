"""Train and save a logistic regression model on the heart disease dataset based on the experiments in initial_modelling.ipynb.
Model: LR with C=1.0, imputer strategy=mean. Outputs a model.joblib file in the root of the project."""

import joblib
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Data Loading
print("Loading data...")
data_path = "./data/heart.csv"
df = pl.read_csv(data_path)
X, y = df[:, :-1], df[:, -1]

missing_values_columns = ["Cholesterol"]
binary_columns = ["FastingBS"]
numeric_columns = [column.name for column in X.select(pl.col(pl.Int64, pl.Float64))]
numeric_columns = [
    column
    for column in numeric_columns
    if column not in binary_columns and column not in missing_values_columns
]
onehot_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina"]
ordinal_columns = ["ST_Slope"]

# Defining pipelines

# pipeline for numeric columns that need imputation and scaling - Cholesterol
impute_and_scale = Pipeline(
    [
        ("imputer", SimpleImputer(missing_values=0, strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "impute_and_scale",
            impute_and_scale,
            missing_values_columns,
        ),
        (
            "categorical_onehot",
            OneHotEncoder(drop="first", sparse_output=False),
            onehot_columns,
        ),
        ("categorical_ordinal", OrdinalEncoder(), ordinal_columns),
        ("scaler", StandardScaler(), numeric_columns),
    ],
    remainder="passthrough",
)
preprocessor.set_output(transform="polars")

lr = Pipeline(
    [
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ]
)

print("Training...")
lr.fit(X, y)  # fit on the entire dataset
print("Training complete")

# save
save_path = "model/model.joblib"
joblib.dump(lr, save_path)
print(f"Model saved to {save_path}")
