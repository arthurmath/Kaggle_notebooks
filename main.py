import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import add_dummy_feature

np.random.seed(42) 




# x input
nb_points = 1000 
X = 2 * np.random.rand(nb_points, 1) 
X_b = add_dummy_feature(X) # add a column (all 1s)


# y input
bias = 4
slope = 3
y = bias + slope * X + np.random.rand(nb_points, 1) 




# Normal Equation
theta_hat = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Slope (a): {theta_hat[0][0]:.2f}  | origin (b):  {theta_hat[1][0]:.2f}")


# Pseudo inverse
theta_hat = np.linalg.pinv(X_b) @ y
print(f"Slope (a): {theta_hat[0][0]:.2f}  | origin (b):  {theta_hat[1][0]:.2f}")


# Scikit learn
model = LinearRegression()
model.fit(X, y)
print(f"Slope (a): {model.intercept_[0]:.2f}  | origin (b):  {model.coef_[0][0]:.2f}")


# SVD
U, S, Vt = np.linalg.svd(X_b, full_matrices=False)
S_inv = np.zeros((S.shape[0], S.shape[0]))
for i in range(len(S)):
    if S[i] != 0:
        S_inv[i, i] = 1 / S[i]
        
# print(f"V shape: {Vt.T.shape} | S shape: {S_inv.shape} | Ut shape: {U.T.shape} | y shape: {y.shape}")
theta_hat_svd = Vt.T @ S_inv @ U.T @ y
print(f"Slope (a): {theta_hat_svd[0][0]:.2f}  | origin (b):  {theta_hat_svd[1][0]:.2f}")



