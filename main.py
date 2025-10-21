import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from plotting.plott import plotting_age_survivers, plotting_class_survivers


data = pd.read_csv('train_and_test2.csv')

X = data[['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass']].values
X_graph = data[['Age', 'Pclass', '2urvived']]
Y = data['2urvived'].values


def logreg(X, Y, learning_rate=0.01, epoches=10000):
    slopes = np.random.randn(X.shape[1])
    intercept = np.random.randn()

    for _ in range(epoches):

        i = np.random.randint(0, len(X))
        x = X[i]
        y = Y[i]

        z = x @ slopes + intercept
        predict = 1 / (1 + np.exp(-z))

        error = predict - y
        slopes -= x * error * learning_rate
        intercept -= error * learning_rate

    return slopes, intercept

def predict(model, X):
    slopes = model[0]
    intercept = model[1]

    z = X @ slopes + intercept
    raw_predictions = 1 / (1 + np.exp(-z))
    predictions = (raw_predictions > 0.5).astype(int)

    return predictions

model = logreg(X, Y)

predictions = predict(model, X)

print('acc: ', (predictions == Y).mean())

plotting_age_survivers(X_graph)
plotting_class_survivers(X_graph)

plt.show()