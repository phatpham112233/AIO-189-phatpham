## question 1 ##
import numpy as np

def compute_mean(X):
    return np.mean(X)

X = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
print("Mean:", compute_mean(X))

## question 2 ##
def compute_median(X):
    size = len(X)
    X = np.sort(X)
    if size % 2 == 0:
        return (X[size // 2 - 1] + X[size // 2]) / 2
    else:
        return X[size // 2]

X = [1, 5, 4, 4, 9, 13]
print("Median:", compute_median(X))

## question 3 ## 

def compute_mean(X):
    return np.mean(X)

def compute_std(X):
    mean = compute_mean(X)
    variance = np.mean([(x - mean) ** 2 for x in X])
    return np.sqrt(variance)

X = [171, 176, 155, 167, 169, 182]
print("Standard Deviation:", compute_std(X))

## question 4 ##

def compute_correlation_cofficient(X, Y):
    N = len(X)
    sum_xy = np.sum([X[i] * Y[i] for i in range(N)])
    sum_x = np.sum(X)
    sum_y = np.sum(Y)
    sum_x2 = np.sum([x ** 2 for x in X])
    sum_y2 = np.sum([y ** 2 for y in Y])

    numerator = N * sum_xy - sum_x * sum_y
    denominator = np.sqrt((N * sum_x2 - sum_x ** 2) * (N * sum_y2 - sum_y ** 2))

    return np.round(numerator / denominator, 2)

X = np.asarray([-2, -5, -11, 6, 4, 15, 9])
Y = np.asarray([4, 25, 121, 36, 16, 225, 81])
print("Correlation:", compute_correlation_cofficient(X, Y))


## 2. TABULAR DATA ANALYSIS ##

import pandas as pd
import numpy as np

def correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

data = pd.read_csv("advertising.csv")
x = data['TV']
y = data['Radio']
corr_xy = correlation(x, y)
print(f"Correlation between TV and Sales: {round(corr_xy, 2)}")
