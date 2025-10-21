import numpy as np
import pandas as pd

from matplotlib import pyplot as plt




def plotting_age_survivers(X):
    BAR_WIDTH = 2
    plt.title('Survive / Age')
    r1 = np.array([15, 25, 40, 55, 70])  # 0-15 16-25 26-40 41-55 56-80

    x_survived = X[X['2urvived'] == True]
    x_notsurvived = X[X['2urvived'] == False]

    survived = [
        x_survived['2urvived'][(x_survived['Age'] <= 15)].count(),
        x_survived['2urvived'][(x_survived['Age'].between(16, 25))].count(),
        x_survived['2urvived'][(x_survived['Age'].between(26, 40))].count(),
        x_survived['2urvived'][(x_survived['Age'].between(41, 55))].count(),
        x_survived['2urvived'][(x_survived['Age'] > 55)].count(),
    ]
    not_survived = [
        x_notsurvived['2urvived'][(x_notsurvived['Age'] <= 15)].count(),
        x_notsurvived['2urvived'][(x_notsurvived['Age'].between(16, 25))].count(),
        x_notsurvived['2urvived'][(x_notsurvived['Age'].between(26, 40))].count(),
        x_notsurvived['2urvived'][(x_notsurvived['Age'].between(41, 55))].count(),
        x_notsurvived['2urvived'][(x_notsurvived['Age'] > 55)].count(),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(r1 - BAR_WIDTH / 2, survived, width=BAR_WIDTH, label='Survived')
    ax.bar(r1 + BAR_WIDTH / 2, not_survived, width=BAR_WIDTH, label='Not survived')

    plt.legend()

    plt.xticks(r1, ['0-15', '16-25', '26-40', '41-55', '56-80'])

def plotting_class_survivers(X):
    plt.title('Survive / Cabin class')
    BAR_WIDTH = 0.3
    # Pclass
    r1 = np.array([1, 2, 3])  # 0-15 16-25 26-40 41-55 56-80

    x_survived = X[X['2urvived'] == 1]
    x_notsurvived = X[X['2urvived'] == 0]

    survived = x_survived.groupby('Pclass')['2urvived'].count()
    not_survived = x_notsurvived.groupby('Pclass')['2urvived'].count()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(r1 - BAR_WIDTH / 2, survived, width=BAR_WIDTH, label='Survived')
    ax.bar(r1 + BAR_WIDTH / 2, not_survived, width=BAR_WIDTH, label='Not survived')

    plt.legend()
    plt.xticks(r1, [1, 2, 3])
