import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


x_train = pd.read_csv(r"..\kaggle_datasets\Titanic\train.csv")
x_test = pd.read_csv(r"..\kaggle_datasets\Titanic\test.csv")

y_train = x_train['Survived']
x_train.drop('Survived', axis=1, inplace=True)

test_id = x_test['PassengerId']
x_train.drop('PassengerId', axis=1, inplace=True)
x_test.drop('PassengerId', axis=1, inplace=True)

features = [
    feature for feature in x_train
    if x_train[feature].nunique() <= 10 and
    x_train[feature].isna().mean() < 0.2
]

x_train = x_train[features]
x_test = x_test[features]


ensemble_model = make_pipeline(
    make_column_transformer(
        (make_pipeline(SimpleImputer(strategy='most_frequent'),
         OneHotEncoder(handle_unknown='ignore')),
         features)
    ),
    VotingClassifier(estimators=[
        ('KNN', KNeighborsClassifier()),
        ('DC', DecisionTreeClassifier()),
        ('RF', RandomForestClassifier()),
        ('GB', GradientBoostingClassifier()),
        ('ADA', AdaBoostClassifier()),
        ('XGB', XGBClassifier()),
        ('LGBM', LGBMClassifier())
    ],
                     voting='soft')
)

ensemble_model.fit(x_train, y_train)

y_pred = ensemble_model.predict(x_test)

submission = pd.DataFrame({'PassengerId': test_id, 'Survived': y_pred})
submission.to_csv(r"\Titanic.csv", index = False)

# Current best score : 0.77751
