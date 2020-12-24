import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv(r"..\kaggle_datasets\Natural Disaster\train.csv")
x_test = pd.read_csv(r"..\kaggle_datasets\Natural Disaster\test.csv")

x_train, y_train = train.drop('target', axis=1), train['target']

test_id = x_test['id']
x_train.drop('id', axis=1, inplace=True)
x_test.drop('id', axis=1, inplace=True)

x_train.drop('location', axis=1, inplace=True)
x_test.drop('location', axis=1, inplace=True)

x_train.drop('text', axis=1, inplace=True)
x_test.drop('text', axis=1, inplace=True)

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

preprocessor = make_pipeline(
    SimpleImputer(missing_values=float('nan'), strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

from xgboost import XGBClassifier

model = make_pipeline(
    preprocessor,
    XGBClassifier()
)

model.fit(x_train, y_train)
print(model.score(x_train, y_train))

y_pred = model.predict(x_test)

submission = pd.DataFrame({"id": test_id, "target": y_pred})
submission.to_csv(r"..\saved\Natural Disaster.csv", index=False)

# Current best score : 0.71345
