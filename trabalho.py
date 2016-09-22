import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

train = pd.read_csv('train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

test = pd.read_csv('test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train, y_train)
y_pred = KNN.predict_proba(x_test)

submission = pd.DataFrame(y_pred, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')