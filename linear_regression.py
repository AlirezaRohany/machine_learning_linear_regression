import tensorflow
import keras
import pandas
import numpy
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# print("yohohoh")

data = pandas.read_csv("student-mat.csv", sep=";")
print(data.head())

# choosing some good attributes for prediction
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

print(data.head(),"\n")

goal_feature = "G3"

X = numpy.array(data.drop([goal_feature], 1))
Y = numpy.array(data[goal_feature])

x_train, x_test, y_train, y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

# print(x_test)
# print(y_test)

# linear regression
linear=linear_model.LinearRegression()
linear.fit(x_train,y_train)
accuracy=linear.score(x_test,y_test)

print("Accuracy: ",accuracy, "\n")
print("Coefficient: \n", linear.coef_,"\n")
print("Intercept: \n",linear.intercept_,"\n")

predictions= linear.predict(x_test)

for i in range(len(predictions)) :
    print("Model predict:", predictions[i] , "    Real value: " , y_test[i])
    print(x_test[i])