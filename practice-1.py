import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu
X = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 0.364,0.398, 0.4, 0.409, 0.421, 0.432, 
              0.473, 0.509, 0.529, 0.561, 0.569, 0.594, 0.638, 0.656, 0.816, 0.853, 0.938, 1.036, 1.045])

y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1,0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
# Khởi tạo dữ liệu
np.random.seed(0)
theta0 = np.random.rand()
theta1 = np.random.rand()
learning_rate = 1e-6
iterations = 1000
# Logistic Function
def logistic_function(z):
    return 1 / (1 + np.exp(-z))
# Prediction Function
def predict(X, theta0, theta1):
    z = theta0 + theta1 * X
    gz = logistic_function(z)
    return gz
# Cost Function (Binary Cross-Entropy)
def cost_function(X, y_true, theta0, theta1):
    m = len(X)
    epsilon = 1e-15
    y_pred = predict(X, theta0, theta1)
    cost = -(1 / m) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return cost
# Gradient Descent
def gradient_descent(X, y, theta0, theta1, learning_rate):
    m = len(X)
    gradient0 = (1 / m) * np.sum(predict(X, theta0, theta1) - y)
    gradient1 = (1 / m) * np.sum((predict(X, theta0, theta1) - y) * X)
    theta0 = theta0 - learning_rate * gradient0
    theta1 = theta1 - learning_rate * gradient1
    return theta0, theta1
# Huấn luyện mô hình
for i in range(iterations):
    theta0, theta1 = gradient_descent(X, y, theta0, theta1, learning_rate)
    cost = cost_function(X, y, theta0, theta1)
print(f'Cost = {cost}, theta0 = {theta0}, theta1 = {theta1}')