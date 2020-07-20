import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from scipy.stats import mode


x_train = pd.read_csv(r"C:\Users\Steva\OneDrive\Desktop\Programming\kaggle_datasets\Titanic\train.csv")
x_test = pd.read_csv(r"C:\Users\Steva\OneDrive\Desktop\Programming\kaggle_datasets\Titanic\test.csv")

y_train = x_train['Survived']
x_train.drop('Survived', axis=1, inplace=True)

test_id = x_test['PassengerId']
x_train.drop('PassengerId', axis=1, inplace=True)
x_test.drop('PassengerId', axis=1, inplace=True)

numerical_features = [feature for feature in x_train
                      if x_train[feature].dtype != 'object']
categorical_features = [feature for feature in x_train
                        if x_train[feature].dtype == 'object' and
                        x_train[feature].isna().mean() < 0.2 and
                        x_train[feature].nunique() < 10]

x_train = x_train[numerical_features + categorical_features]
x_test = x_test[numerical_features + categorical_features]


preprocessor = make_column_transformer(
    (SimpleImputer(strategy='mean'), numerical_features),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')), categorical_features)
)

models = [make_pipeline(preprocessor, model)
          for model in [RandomForestClassifier(),
                        GradientBoostingClassifier(),
                        XGBClassifier(),
                        LGBMClassifier()]]

for model in models:
    model.fit(x_train, y_train)

y_pred = mode(np.column_stack([model.predict(x_test) for model in models]), axis=1)[0].reshape(-1)

submission = pd.DataFrame({'PassengerId': test_id, 'Survived': y_pred})
submission.to_csv(r"C:\Users\Steva\OneDrive\Desktop\Programming\saved\Titanic.csv", index = False)

# Current best score : 0.76794
