import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import optuna
from optuna.samplers import TPESampler


def objective(trial):
    df = pd.read_csv("data/train.csv")
    df.drop(columns=["Id"], inplace=True)
    col_numeric = df.drop(columns="SalePrice").select_dtypes(include=np.number).columns
    col_categorical = df.select_dtypes(include="object").columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("select", SelectPercentile(percentile=50)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, col_numeric),
            ("cat", categorical_transformer, col_categorical),
        ]
    )

    xgbmodel = XGBRegressor()
    regr = Pipeline(steps=[("preprocessor", preprocessor), ("xgbmodel", xgbmodel)])

    final_regr = TransformedTargetRegressor(
        regressor=regr, transformer=QuantileTransformer(output_distribution="normal")
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["SalePrice"]), df["SalePrice"], test_size=0.2, random_state=42
    )

    params = {
        "xgbmodel__max_depth": trial.suggest_int("max_depth", 1, 9),
        "xgbmodel__learning_rate": trial.suggest_float("learning_rate", 0.001, 1.0),
        "xgbmodel__n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "xgbmodel__min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "xgbmodel__gamma": trial.suggest_float("gamma", 1e-8, 1.0),
        "xgbmodel__subsample": trial.suggest_float("subsample", 0.01, 1.0),
        "xgbmodel__colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.01, 1.0
        ),
        "xgbmodel__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0),
        "xgbmodel__reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0),
        "xgbmodel__use_label_encoder": False,
    }
    regr.set_params(**params)
    final_regr.fit(X_train, y_train)

    return final_regr.score(X_test, y_test)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
study.best_params


df = pd.read_csv("data/train.csv")
df.drop(columns=["Id"], inplace=True)
col_numeric = df.drop(columns="SalePrice").select_dtypes(include=np.number).columns
col_categorical = df.select_dtypes(include="object").columns

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ("select", SelectPercentile(percentile=50)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, col_numeric),
        ("cat", categorical_transformer, col_categorical),
    ]
)

xgbmodel = XGBRegressor()
regr = Pipeline(steps=[("preprocessor", preprocessor), ("xgbmodel", xgbmodel)])

final_regr = TransformedTargetRegressor(
    regressor=regr, transformer=QuantileTransformer(output_distribution="normal")
)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["SalePrice"]), df["SalePrice"], test_size=0.2, random_state=42
)


best_params = {"xgbmodel__" + k: v for k, v in study.best_params.items()}


regr.set_params(**best_params)
final_regr.fit(X_train, y_train)
final_regr.score(X_test, y_test)


test = pd.read_csv("data/test.csv")
sales_prediction = final_regr.predict(test.drop(columns=["Id"]))

submission = pd.DataFrame({"Id": test["Id"], "SalePrice": sales_prediction})
submission.to_csv("data/submission_02_optuna2.csv", index=False)
