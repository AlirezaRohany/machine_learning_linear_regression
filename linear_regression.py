import tensorflow
import keras
import pandas
import numpy
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib.pyplot import style
import pickle

# print("yohohoh")

data = pandas.read_csv("student-mat.csv", sep=";")
print(data.head())

# choosing some good attributes for prediction
selected_features = ["G1", "G2", "G3", "studytime", "failures", "absences"]
data = data[selected_features]

print(data.head(), "\n")

goal_feature = "G3"

X = numpy.array(data.drop([goal_feature], 1))
Y = numpy.array(data[goal_feature])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# print(x_test)
# print(y_test)

best_model_accuracy = 0
for i in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    # linear regression
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)
    print("Accuracy: ", accuracy, "\n")

    # save model
    if accuracy > best_model_accuracy:
        best_model_accuracy = accuracy
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
print("Best accuracy: ", best_model_accuracy)

# use model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficient: \n", linear.coef_, "\n")
print("Intercept: \n", linear.intercept_, "\n")

predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print("Model predict:", predictions[i], "    Real value: ", y_test[i])
    print(x_test[i])

# plotting all data
style.use("ggplot")

for feature in selected_features:
    if feature != "G3":
        pyplot.scatter(data[feature], data[goal_feature], color="red")
        pyplot.xlabel(feature)
        pyplot.ylabel("Final grade")
        pyplot.title("All data")
        pyplot.show()

# plotting test data vs predictions
for j in range(5):
    for i in range(len(predictions)):
        pyplot.scatter(x_test[i][j], y_test[i], color="blue")
        pyplot.scatter(x_test[i][j], predictions[i], color="orange")
    if j < 2:
        pyplot.xlabel(selected_features[j])
    else:
        pyplot.xlabel(selected_features[j + 1])
    pyplot.ylabel("Final grade")
    pyplot.title("Test data(blue) vs Predictions(orange)")
    pyplot.show()
