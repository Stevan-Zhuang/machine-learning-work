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

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
train_vector = count_vectorizer.fit_transform(x_train['text'])
test_vector = count_vectorizer.transform(x_test['text'])

from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(train_vector, y_train)

y_pred = model.predict(test_vector)

submission = pd.DataFrame({"id": test_id, "target": y_pred})
submission.to_csv(r"..\saved\Natural Disaster.csv", index=False)

# Current best score : 0.78148
