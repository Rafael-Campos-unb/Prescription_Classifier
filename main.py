import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

file = pd.read_csv("drug200.csv")

categories = {'HIGH': 0, 'NORMAL': 1, 'LOW': 2}
categories_dg = {'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugY': 3, 'drugX': 4}
categories_sex = {'F': 0, 'M': 1}
list_BP = []
list_Ch = []
list_Dg = []
list_Sx = []


def categories_map(data, word, list_):
    for value in data[word]:
        list_.append(value)
        for i in range(len(list_)):
            if list_[i] == 'HIGH':
                list_[i] = categories['HIGH']
            if list_[i] == 'NORMAL':
                list_[i] = categories['NORMAL']
            if list_[i] == 'LOW':
                list_[i] = categories['LOW']
    file[word] = list_


def drugs_map(data, word, list_):
    for value in data[word]:
        list_.append(value)
        for i in range(len(list_)):
            if list_[i] == 'drugA':
                list_[i] = categories_dg['drugA']
            if list_[i] == 'drugB':
                list_[i] = categories_dg['drugB']
            if list_[i] == 'drugC':
                list_[i] = categories_dg['drugC']
            if list_[i] == 'drugY':
                list_[i] = categories_dg['drugY']
            if list_[i] == 'drugX':
                list_[i] = categories_dg['drugX']
    file[word] = list_


def sex_map(data, word, list_):
    for value in data[word]:
        list_.append(value)
        for i in range(len(list_)):
            if list_[i] == 'F':
                list_[i] = categories_sex['F']
            if list_[i] == 'M':
                list_[i] = categories_sex['M']
    file[word] = list_


categories_map(file, 'BP', list_BP)
categories_map(file, 'Cholesterol', list_Ch)
drugs_map(file, 'Drug', list_Dg)
sex_map(file, 'Sex', list_Sx)

X = file.drop('Drug', axis=1)
X = X.to_numpy()
X = preprocessing.StandardScaler().fit_transform(X)

y = list_Dg

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=categories_dg.keys(), yticklabels=categories_dg.keys(),
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.savefig('graph.png')

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='macro')
recall = metrics.recall_score(y_test, y_pred, average='macro')

print(f"Accuracy:{accuracy*100:.2f}")
print(f"Precision:{ precision*100:.2f}")
print(f"Recall:{ recall*100:.2f}")
