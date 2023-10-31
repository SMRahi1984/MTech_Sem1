# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading data
insurancedata=pd.read_csv('/content/sample_data/insurance.csv')

# Converting categorical to integers
# The pd.get_dummies() function creates a new DataFrame with binary columns for each unique value in the specified column(s).
# The drop_first=True argument drops the first column to avoid multicollinearity in regression models.
insurancedata_df = pd.get_dummies(insurancedata,columns=['sex','smoker','region'],drop_first=True)
#insurancedata_df.head()

X = insurancedata["bmi"]
y = insurancedata["charges"]

# plt.scatter(X, y)
# plt.xlabel("BMI")
# plt.ylabel("Insurance Charges")
# plt.title("Scatter Plot of BMI vs. Insurance Charges")
# plt.show()

# Linear regression method
# y=c+mx
def linear_regression(X, theta0, theta1):
    return theta0 + theta1 * X 

#define cost method (Mean squared error)
def cost_function(X, y, theta0, theta1):
    m = len(X)
    predictions = linear_regression(X, theta0, theta1)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost    

#define gradient descent function
def gradient_descent(X, y, theta0, theta1, learning_rate, iterations):
    m = len(X)
    cost_history = []

    for _ in range(iterations):
        predictions = linear_regression(X, theta0, theta1)
        error = predictions - y
        theta0 -= (learning_rate / m) * np.sum(error)
        theta1 -= (learning_rate / m) * np.sum(error * X)
        cost = cost_function(X, y, theta0, theta1)
        cost_history.append(cost)

    return theta0, theta1, cost_history

# Train the model with gradient descent
learning_rate = 0.000001
iterations = 1000
initial_theta0 = 0
initial_theta1 = 0
theta0, theta1, cost_history = gradient_descent(X, y, initial_theta0, initial_theta1, learning_rate, iterations)

# Step 8: Visualize the regression line along with the data points
plt.scatter(X, y, label="Data")
plt.plot(X, linear_regression(X, theta0, theta1), color="red", label="Regression Line")
plt.xlabel("BMI")
plt.ylabel("Insurance Charges")
plt.title("Linear Regression with Gradient Descent")
plt.legend()
plt.show()

print("Optimal Theta0:", theta0)
print("Optimal Theta1:", theta1)