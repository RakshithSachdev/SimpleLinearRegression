from math import sqrt

def costfunc(actual, predic):
    sum_error = 0.0
    for i in range(len(actual)):
        predic_err = predic[i] - actual[i]
        sum_error += (predic_err ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

def predict_values(data, algo):
    test_set = list()
    for r in data:
        row_copy = list(r)
        row_copy[-1] = None
        test_set.append(row_copy)
    predic = algo(data, test_set)
    print(predic)
    actual = [r[-1] for r in data]
    rmse = costfunc(actual, predic)
    return rmse

def mean(values):
    return sum(values) / float(len(values))

def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

def coefficients(data):
    x = [row[0] for row in data]
    y = [row[1] for row in data]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

def linear_regression(train, test):
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        hyp = b0 + b1*row[0]
        predictions.append(hyp)
    return predictions

data = [[1,2], [2, 3], [4, 5], [6, 6], [8, 9]]
rmserror = predict_values(data, linear_regression)
print('Error in prediction: %.2f'%(rmserror))
