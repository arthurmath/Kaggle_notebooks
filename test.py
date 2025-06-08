import numpy as np
import time
import math
import os
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


model = LinearRegression()



n = 50
A = np.ones((n, n))
print(A.shape)

U = A[:, 2].reshape((n, 1))
V = A[2, :].reshape((1, n))

print(U.shape, V.shape)

Ai = U @ V
Ak = V @ U

print(Ai.shape)
print(Ak.shape)

