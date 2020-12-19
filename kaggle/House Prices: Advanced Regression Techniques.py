import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


x_train = pd.read_csv(r"..\kaggle_datasets\train.csv")
x_test = pd.read_csv(r"..\kaggle_datasets\test.csv")

y_train = x_train["SalePrice"]
x_train.drop("Id", axis=1, inplace=True)
x_train.drop("SalePrice", axis=1, inplace=True)

test_id = x_test["Id"]
x_test.drop("Id", axis=1, inplace=True)

numerical_features = [feature for feature in x_train
                      if x_train[feature].dtype != "object"]
categorical_features = [feature for feature in x_train
                        if x_train[feature].dtype == "object" and
                        x_train[feature].isna().mean() < 0.2 and
                        x_train[feature].nunique() < 12]

x_train = x_train[numerical_features + categorical_features]
x_test = x_test[numerical_features + categorical_features]

preprocessor = make_column_transformer(
    (SimpleImputer(strategy="mean"), numerical_features),
    (make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical_features)
)


def make_model(model):
    return make_pipeline(preprocessor, model)

models = [make_model(model) for model in [RandomForestRegressor(),
                                          GradientBoostingRegressor(),
                                          KernelRidge(),
                                          XGBRegressor(),
                                          LGBMRegressor()]]

for model in models:
    model.fit(x_train, y_train)

y_pred = sum(model.predict(x_test) for model in models)/len(models)

submission = pd.DataFrame({"Id": test_id, "SalePrice": y_pred})
submission.to_csv(r"../input/House Prices.csv", index = False)

# Current best score: 0.12952
