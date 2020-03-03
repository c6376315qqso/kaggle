import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

data_train = pd.read_csv('C:\\Users\\Jame Black\\Desktop\\train.csv')
data_test = pd.read_csv('C:\\Users\\Jame Black\\Desktop\\test.csv')
len_train = data_train.shape[0]
# concat train and test data
data_combine = pd.concat([data_train, data_test], ignore_index=True)

# no duplicated data
# print('有无重复数据：' ,any(data_combine.duplicated()))

# add attr title
data_combine['Title'] = data_combine.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
list_title_discard = data_combine.Title.value_counts().index[4:].to_list()
data_combine.Title = data_combine.Title.apply(lambda x: 'Rare' if x in list_title_discard else x)

# fill age with median according to title
ta = data_combine.groupby(by='Title').Age.median()
data_combine.Age = data_combine[['Age', 'Title']].apply(lambda x: ta[x[1]] if pd.isnull(x[0]) else x[0], axis=1)

# add attr has_cabin
data_combine['has_cabin'] = 0
data_combine.loc[~data_combine.Cabin.isnull(), 'has_cabin'] = 1
data_combine.loc[data_combine.Cabin.isnull(), 'has_cabin'] = 0

# plot the number of survivor of has or don't has cabin
# pd.crosstab(data_combine.has_cabin, data_combine.Survived).plot(kind='bar')
# plt.show()

# fill embarked with mode
data_combine.Cabin.fillna('M', inplace=True)
data_combine.Embarked.fillna(data_combine.Embarked.mode()[0], inplace=True)
data_combine.Fare.fillna(data_combine.Fare.mean(), inplace=True)
data_combine.Cabin = data_combine.Cabin.apply(lambda x: x[0])

# ax1 = plt.subplot2grid(shape=(2, 4), loc=(0, 0), rowspan=2,colspan=2)
# pd.crosstab(data_combine.Parch[:len_train], data_train.Survived).plot(kind='bar', ax=ax1)
# ax2 = plt.subplot2grid(shape=(2, 4), loc=(0, 2), rowspan=2,colspan=2)
# pd.crosstab(data_combine.SibSp[:len_train], data_train.Survived).plot(kind='bar', ax=ax2)
# plt.show()

data_combine['Family'] = data_combine[['SibSp', 'Parch']].apply(lambda x: x[0] + x[1] + 1, axis=1)

# pd.crosstab(pd.qcut(data_combine.Fare, 4)[:len_train], data_train.Survived).plot.bar(rot = 360)
# plt.show()
data_combine.drop('Name', inplace=True, axis=1)
data_combine.drop('Ticket', inplace=True, axis=1)
data_combine.Sex = data_combine.Sex.map({'male': 1, 'female': 0})
data_combine.Cabin = data_combine.Cabin.map({'M': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'A': 7, 'T': 7})
data_combine.Embarked = data_combine.Embarked.map(dict(C=0, Q=1, S=2))
data_combine.Title = data_combine.Title.map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})
data_combine.Fare = pd.cut(data_combine.Fare, 6, labels=False)
data_combine.Age = pd.cut(data_combine.Age, 4, labels=False)

cross_line = len_train * 0.8

data_train_x = data_combine.loc[:cross_line, 'Pclass':]
data_train_y = data_combine.loc[:cross_line, 'Survived']

data_cross_x = data_combine.loc[cross_line + 1:len_train - 1, 'Pclass':]
data_cross_y = data_combine.loc[cross_line + 1:len_train - 1, 'Survived']

vec_C = [0.01, 0.05, 0.2, 0.5, 1, 5, 10]

svm_clf = svm.SVC(C=5)
svm_clf.fit(data_train_x, data_train_y)
# print('C = ', i)
print(svm_clf.score(data_train_x, data_train_y))
print(svm_clf.score(data_cross_x, data_cross_y))

data_test_x = data_combine.loc[len_train:, 'Pclass':]
pred = svm_clf.predict(data_test_x).astype(int)
out = pd.DataFrame({'PassengerId': data_combine.loc[len_train:, 'PassengerId'], 'Survived': pred})
out.to_csv('C:\\Users\\Jame Black\\Desktop\\out.csv', index=False)
