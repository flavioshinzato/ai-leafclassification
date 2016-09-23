import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss, f1_score, recall_score

def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)  # encode species strings
    classes = list(le.classes_)  # save column names for submission
    test_ids = test.id  # save test ids for submission

    train = train.drop(['species', 'id'], axis=1)
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes


train_ = pd.read_csv('train.csv')
test_ = pd.read_csv('test.csv')

train, labels, test, test_ids, classes = encode(train_, test_)

sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

KNN = KNeighborsClassifier(n_neighbors=5)

acc = 0
f1 = 0
recall = 0
ll = 0

for train_index, test_index in sss:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    KNN.fit(X_train, y_train)
    train_predictions = KNN.predict(X_test)
    acc = acc + accuracy_score(y_test, train_predictions)
    f1 = f1 + f1_score(y_test, train_predictions, average='macro')
    recall = recall + recall_score(y_test, train_predictions, average='macro')

    train_predictions = KNN.predict_proba(X_test)
    ll = ll + log_loss(y_test, train_predictions)

print "Accuracy: %f" % (acc/10)
print "F1: %f" % (f1/10)
print "Recall: %f" % (recall/10)
print "Log-loss: %f" % (ll/10)

test_predictions = KNN.predict_proba(test)

# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('submission.csv', index = False)
submission.tail()