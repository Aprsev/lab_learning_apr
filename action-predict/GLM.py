import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Create a synthetic dataset
np.random.seed(42)
n_samples = 100
X = np.random.rand(n_samples, 2)  # Two independent variables
# Simulate a Poisson-distributed response variable
lambda_values = np.exp(1 + 2 * X[:, 0] - 1 * X[:, 1])
y = np.random.poisson(lambda_values)

# Create a DataFrame for the dataset
data = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'Y': y})

# Fit a Poisson regression model using GLM
X = sm.add_constant(X)  # Add an intercept term
model = sm.GLM(y, X, family=sm.families.Poisson())
result = model.fit()

# Print the summary of the GLM
print(result.summary())

# Plot the observed vs. predicted values
predicted = result.predict(X)
plt.scatter(y, predicted)
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.title("Observed vs. Predicted Values")
plt.show()
