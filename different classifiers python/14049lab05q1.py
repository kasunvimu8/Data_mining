import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import neighbors , datasets
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#problem 1
fileName = 'zooData.csv'
data = pd.read_csv(fileName)
print("* Problem 1 : Printing the zoo data set.... \n")
print(data)
data.drop(['animalName'], axis=1, inplace=True)
Y = data['type']
data.drop(['type'], axis=1, inplace=True)
X = data

#problem 2
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
print("* Problem 2 :\n")
print("classification accuracy using decision tree classifier by training set... ")
print('Training Accuracy: ',clf.score(X,Y)) # Training accuracy

X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size =0.333,random_state =0)
clf.fit(X_train ,Y_train) # clf is a classifier.
print("\nclassification accuracy using decision tree classifier by splitting the data as 2/3 for training and 1/3 for testing... ")
print('Test Accuracy: ',clf.score(X_test ,Y_test )) # Test accuracy

# 10-fold cross validation
scores = cross_val_score(clf, X, Y, cv=10) # 10-fold cross validation
print("\nclassification accuracy using decision tree classifier by 10-fold cross validation... ")
print(scores) # Results for all the folds
print("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

# 10 fold cross validation provides more realistic future performance as it uses the whole test set as trainning set and also as the test set in each fold
print("\nAnswer  : 10 fold cross validation provides more realistic future performance as it uses the whole test set as trainning set and also as the test set in each fold")

 #10 fold cross validation provides more realistic future performance as it uses the whole test set as trainnig
 #set and also as the test set in each fold and another issue is that training set may be irralavent in future
 #with changing environment.Therefore even the traning set have the highest accuracy it may be useless for future predictions.

#problem 3

# it doesn't make any sense of having confusion matrix in trainning set where accuracy is 1.0 (diagonal matrix)
print("\n* Problem 3 :\n")
print("it doesn't make any sense of having confusion matrix in trainning set where accuracy is 1.0 (diagonal matrix)\n")

# by splitting the data as 2/3 for training and 1/3 for testing
clf = tree.DecisionTreeClassifier()
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size =0.333,random_state =0)
clf.fit(X_train ,Y_train) # clf is a classifier.
Y_pred = clf.fit(X_train , Y_train ).predict(X_test)
print("\nConfusion matrix using Decision tree classifier..")
print(confusion_matrix(Y_test , Y_pred ))
# in above confusion matrix 4 classification errors have happend. (7 classified as 1)
print("in above confusion matrix 4 classification errors have happend. (7 classified as 1)\n")

clf = GaussianNB()
clf.fit(X,Y)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size =0.333,random_state =0)
clf.fit(X_train ,Y_train) # clf is a classifier.
Y_pred = clf.fit(X_train , Y_train ).predict(X_test)
print("\nConfusion matrix using GaussianNB..")
print(confusion_matrix(Y_test , Y_pred ))
# in above confusion matrix 4 classification errors have happend. (7 classified as 1, 5 & 6) 
print("in above confusion matrix 4 classification errors have happend. (7 classified as 1, 5 & 6)\n")

clf=MultinomialNB() # clf is a classifier.
clf.fit(X,Y)
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size =0.333,random_state =0)
clf.fit(X_train ,Y_train) # clf is a classifier.
Y_pred = clf.fit(X_train , Y_train ).predict(X_test)
print("\nConfusion matrix using MultinomialNB..")
print(confusion_matrix(Y_test , Y_pred ))
# in above confusion matrix 5 classification errors have happend. (7 classified as 2 & 6, 1 classified as 5) 
print("in above confusion matrix 5 classification errors have happend. (7 classified as 2 & 6, 1 classified as 5) \n")

clf = neighbors.KNeighborsClassifier(n_neighbors =1) # 1-Nearest Neighbor
clf.fit(X,Y) # Model
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size =0.333,random_state =0)
clf.fit(X_train ,Y_train) # clf is a classifier.
Y_pred = clf.fit(X_train , Y_train ).predict(X_test)
print("\nConfusion matrix using KNeighbors Classifier..")
print(confusion_matrix(Y_test , Y_pred ))
# in above confusion matrix 4 classification errors have happend. (7 classified as 1 & 3) 
print("in above confusion matrix 4 classification errors have happend. (7 classified as 1 & 3) \n")

clf = svm.SVC(kernel='linear', C=1,gamma=1).fit(X, Y) # clf is a classifier
clf.fit(X,Y) # Model
X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size =0.333,random_state =0)
clf.fit(X_train ,Y_train) # clf is a classifier.
Y_pred = clf.fit(X_train , Y_train ).predict(X_test)
print("\nConfusion matrix using SVM..")
print(confusion_matrix(Y_test , Y_pred ))
# in above confusion matrix 4 classification errors have happend. (7 classified as 1 & 3) 
print("in above confusion matrix 4 classification errors have happend. (7 classified as 1 & 3) \n")

print("\n* Problem 4 :\n")
clf = tree.DecisionTreeClassifier()
clf.fit(X,Y)
scores = cross_val_score(clf, X, Y, cv=10) # 10-fold cross validation
print("10-fold cross validation accuracy of Decision Tree Classifier\n")
print(scores) # Results for all the folds
print("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

clf = GaussianNB()
clf.fit(X,Y)
scores = cross_val_score(clf, X, Y, cv=10) # 10-fold cross validation
print("\n\n10-fold cross validation accuracy of GaussianNB\n")
print(scores) # Results for all the folds
print("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

clf=MultinomialNB() # clf is a classifier.
clf.fit(X,Y)
scores = cross_val_score(clf, X, Y, cv=10) # 10-fold cross validation
print("\n\n10-fold cross validation accuracy of MultinomialNB\n")
print(scores) # Results for all the folds
print("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

clf = neighbors.KNeighborsClassifier(n_neighbors =1) # 1-Nearest Neighbor
clf.fit(X,Y) # Model
scores = cross_val_score(clf, X, Y, cv=10) # 10-fold cross validation
print("\n\n10-fold cross validation accuracy of KNeighborsClassifier\n")
print(scores) # Results for all the folds
print("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

clf = svm.SVC(kernel='linear', C=1,gamma=1).fit(X, Y) # clf is a classifier
clf.fit(X,Y) # Model
scores = cross_val_score(clf, X, Y, cv=10) # 10-fold cross validation
print("\n\n10-fold cross validation accuracy of SVM\n")
print(scores) # Results for all the folds
print("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))


# Problem 5
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X,Y) # Model
scores = cross_val_score(clf, X, Y, cv=10) # 10-fold cross validation
print("\n\n* Problem 5 :")
print("\n\n10-fold cross validation accuracy of RandomForestClassifier\n")
print(scores) # Results for all the folds
print("10CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std()*2))

X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size =0.333,random_state =0)
clf.fit(X_train ,Y_train) # clf is a classifier.
Y_pred = clf.fit(X_train , Y_train ).predict(X_test)
print("\nConfusion matrix using RandomForestClassifier..")
print(confusion_matrix(Y_test , Y_pred ))

#it has good accuracy like 0.96 compared with other classifications. 
#KNeighborsClassifier has the best accuracy.