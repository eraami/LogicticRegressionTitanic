import pandas as pd
import numpy as np

from sklearn import linear_model

from matplotlib import pyplot as plt
from plotting.plott import plotting_age_survivers, plotting_class_survivers


data = pd.read_csv('train_and_test2.csv')

X = data[['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass']].values
X_graph = data[['Age', 'Pclass', '2urvived']]
Y = data['2urvived'].values

age = X[:, 0]
X[:, 0] = (age - age.min()) / (age.max() -  age.min())

fare = X[:, 1]
X[:, 1] = (fare - fare.min()) / (fare.max() -  fare.min())

plt.scatter(X_graph['Age'], X[:, 1])
plt.show()

def logreg(X, Y, learning_rate=0.001, epoches=10000):
    slopes = np.random.randn(X.shape[1])
    intercept = np.random.randn()

    BATCH_SIZE = 16
    BATCH_INDEXES = np.array(np.arange(len(X)))

    for _ in range(epoches):
        np.random.shuffle(BATCH_INDEXES)
        x = X[BATCH_INDEXES[:BATCH_SIZE]]
        y = Y[BATCH_INDEXES[:BATCH_SIZE]]
        z = x @ slopes + intercept
        predict = 1 / (1 + np.exp(-z))
        error = predict - y
        gradient_slope = 1 / BATCH_SIZE * x.T @ error
        gradient_intercept = error.mean()
        slopes -= gradient_slope * learning_rate
        intercept -= gradient_intercept * learning_rate

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
sk_model = linear_model.LogisticRegression()
sk_model.fit(X, Y)
sk_predictions = sk_model.predict(X)

print('my acc: ', (predictions == Y).mean())
print('sk acc: ', (sk_predictions == Y).mean())
print('my/sk acc: ', (predictions == sk_predictions).mean())
#
# plotting_age_survivers(X_graph)
# plt.show()
# plotting_class_survivers(X_graph)
# plt.show()

