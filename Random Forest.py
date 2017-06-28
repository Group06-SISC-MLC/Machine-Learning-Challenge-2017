# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:07:58 2017

@author: njc
"""
from timeit import default_timer as timer
start = timer()
print("hello")
        
from sklearn import tree   
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)
        
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)
        
predictions = my_classifier.predict(X_test) 

from sklearn.externals.six import StringIO
dot_data = StringIO()
tree.export_graphviz(my_classifier,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)
print("number ")
        
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)

my_classifier0 = tree.DecisionTreeClassifier()
my_classifier0.fit(X_train, y_train)
        
predictions = my_classifier0.predict(X_test)

from sklearn.externals.six import StringIO
dot_data0 = StringIO()
tree.export_graphviz(my_classifier0,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)
print("number 0")
        
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)
        
my_classifier1 = tree.DecisionTreeClassifier()
my_classifier1.fit(X_train, y_train)
        
predictions = my_classifier1.predict(X_test)  

from sklearn.externals.six import StringIO
dot_data1 = StringIO()
tree.export_graphviz(my_classifier1,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)
print("number 1")  
    

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn import tree    
my_classifier2 = tree.DecisionTreeClassifier()
my_classifier2.fit(X_train, y_train)

predictions = my_classifier2.predict(X_test)

from sklearn.externals.six import StringIO
dot_data2 = StringIO()
tree.export_graphviz(my_classifier2,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)
print("number 2")

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)
        
my_classifier3 = tree.DecisionTreeClassifier()
my_classifier3.fit(X_train, y_train)

predictions = my_classifier3.predict(X_test)


from sklearn.externals.six import StringIO
dot_data3 = StringIO()
tree.export_graphviz(my_classifier3,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)
print("number 3")
        
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn import tree    
my_classifier4 = tree.DecisionTreeClassifier()
my_classifier4.fit(X_train, y_train)
        
predictions = my_classifier4.predict(X_test)

from sklearn.externals.six import StringIO
dot_data4 = StringIO()
tree.export_graphviz(my_classifier4,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)
print("number 4")


x = 0
x0 = 0
x1 = 0
x2 = 0
x3 = 0
x4 = 0


if dot_data.getvalue() == dot_data0.getvalue():
    x = x + 1
if dot_data.getvalue() == dot_data1.getvalue():
    x = x + 1
if dot_data.getvalue() == dot_data2.getvalue():
    x = x + 1
if dot_data.getvalue() == dot_data3.getvalue():
    x = x + 1
if dot_data.getvalue() == dot_data4.getvalue():
    x = x + 1
if dot_data.getvalue() == dot_data.getvalue():
    x = x + 1

if dot_data0.getvalue() == dot_data0.getvalue():
    x0 = x0 + 1
if dot_data0.getvalue() == dot_data1.getvalue():
    x0 = x0 + 1
if dot_data0.getvalue() == dot_data2.getvalue():
    x0 = x0 + 1
if dot_data0.getvalue() == dot_data3.getvalue():
    x0 = x0 + 1
if dot_data0.getvalue() == dot_data4.getvalue():
    x0 = x0 + 1
if dot_data0.getvalue() == dot_data.getvalue():
    x0 = x0 + 1
    
if dot_data1.getvalue() == dot_data0.getvalue():
    x1 = x1 + 1
if dot_data1.getvalue() == dot_data1.getvalue():
    x1 = x1 + 1
if dot_data1.getvalue() == dot_data2.getvalue():
    x1 = x1 + 1
if dot_data1.getvalue() == dot_data3.getvalue():
    x1 = x1 + 1
if dot_data1.getvalue() == dot_data4.getvalue():
    x1 = x1 + 1
if dot_data1.getvalue() == dot_data.getvalue():
    x1 = x1 + 1

if dot_data2.getvalue() == dot_data0.getvalue():
    x2 = x2 + 1
if dot_data2.getvalue() == dot_data1.getvalue():
    x2 = x2 + 1
if dot_data2.getvalue() == dot_data2.getvalue():
    x2 = x2 + 1
if dot_data2.getvalue() == dot_data3.getvalue():
    x2 = x2 + 1
if dot_data2.getvalue() == dot_data4.getvalue():
    x2 = x2 + 1
if dot_data2.getvalue() == dot_data.getvalue():
    x2 = x2 + 1
    
if dot_data3.getvalue() == dot_data0.getvalue():
    x3 = x3 + 1
if dot_data3.getvalue() == dot_data1.getvalue():
    x3 = x3 + 1
if dot_data3.getvalue() == dot_data2.getvalue():
    x3 = x3 + 1
if dot_data3.getvalue() == dot_data3.getvalue():
    x3 = x3 + 1
if dot_data3.getvalue() == dot_data4.getvalue():
    x3 = x3 + 1
if dot_data3.getvalue() == dot_data.getvalue():
    x3 = x3 + 1
    
if dot_data4.getvalue() == dot_data0.getvalue():
    x4 = x4 + 1
if dot_data4.getvalue() == dot_data1.getvalue():
    x4 = x4 + 1
if dot_data4.getvalue() == dot_data2.getvalue():
    x4 = x4 + 1
if dot_data4.getvalue() == dot_data3.getvalue():
    x4 = x4 + 1
if dot_data4.getvalue() == dot_data4.getvalue():
    x4 = x4 + 1
if dot_data4.getvalue() == dot_data.getvalue():
    x4 = x4 + 1
    
print(x, x0, x1, x2, x3, x4)

list = [x, x0, x1, x2, x3, x4]

data = {
        'x' : x,
        'x0': x0,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4}

y = key=data.get
print(max(data, key=data.get), " has been voted the most accurate classifier")
if y == 'x':
    print(dot_data.getvalue())
if y == 'x0':
    print(dot_data.getvalue())
if y == 'x1':
    print(dot_data.getvalue())
if y == 'x2':
    print(dot_data.getvalue())
if y == 'x3':
    print(dot_data.getvalue())
if y == 'x4':
    print(dot_data.getvalue())
    
print(max(data, key=data.get))


from sklearn.metrics import accuracy_score
treescore2 = accuracy_score(y_test, predictions)
print(treescore2)
        
print("final accuracy:", treescore2)
end = timer()
print(end - start, "Milliseconds")



