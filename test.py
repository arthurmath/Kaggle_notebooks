import numpy as np
import time
import math
import os
import scipy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV, Lasso, SGDRegressor, ElasticNet, MultiTaskElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor




## Reproducability
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)









# Physical constants
G = 9.81  # gravity (m/s²)
AIR_DENSITY = 1.225  # kg/m³ (air density at sea level)


# Rocket parameters
drag_coefficient = 0.5    # simplified drag coefficient
rocket_area = 1.0         # cross-sectional area (m²)
initial_mass = 500.0      # kg (total mass at liftoff)
fuel_mass = 300.0         # kg of fuel
burn_rate = 2.0           # kg/s (fuel consumption rate)
thrust = 5000.0           # N (constant thrust while fuel remains)




def rocket_simulation(rocket_area, initial_mass, fuel_mass, burn_rate, thrust, drag_coefficient):
    """ Simulates a 1D rocket launch and descent with constant thrust and drag. """

    # Initial conditions
    altitude = 0.1
    velocity = 0.0
    mass = initial_mass
    time = 0.0
    dt = 1  # time step (s)
    t_max = 300  # simulation duration (s)
    
    # State arrays
    altitudes = []
    velocities = []
    accelerations = []
    masses = []
    times = []

    # Simulation loop
    for time in np.linspace(0, t_max, int(t_max / dt)):
        
        # Update mass based on fuel consumption
        if fuel_mass > 0:
            dm = burn_rate * dt
            if dm > fuel_mass:
                dm = fuel_mass  # consume remaining fuel only
            fuel_mass -= dm
            mass -= dm
            current_thrust = thrust
        else:
            current_thrust = 0

        # Forces
        weight = mass * G
        drag = 0.5 * rocket_area * drag_coefficient * AIR_DENSITY * velocity**2 * np.sign(velocity)
        net_force = current_thrust - weight - drag
        acceleration = net_force / mass

        # Euler integration
        velocity += acceleration * dt
        altitude += velocity * dt

        # Store state
        times.append(time)
        altitudes.append(max(altitude, 0))  # avoid negative altitude display
        velocities.append(velocity)
        accelerations.append(acceleration)
        masses.append(mass)
        
    return times, altitudes, velocities, masses






def generate_random_params():
    rocket_area = np.random.uniform(0.5, 2.0)
    initial_mass = np.random.uniform(300, 800)
    fuel_mass = np.random.uniform(0.1, 0.7) * initial_mass 
    burn_rate = np.random.uniform(1, 10)
    thrust = np.random.uniform(100, 1000)
    drag_coefficient = np.random.uniform(0.1, 1.0)
    
    return [rocket_area, initial_mass, fuel_mass, burn_rate, thrust, drag_coefficient]


X = []
Y = []
for i in range(1000):
    params = generate_random_params()
    _, altitudes, _, _ = rocket_simulation(*params)
        
    if np.isnan(altitudes).any():
        print("Simulation failed due to NaN altitude.")
        print("Params : ", params)
        print("itération : ", i)
        print("Altitudes : ", altitudes)
        break
        
    X.append(params)
    Y.append(altitudes)

# X = np.array(X)
# Y = np.array(Y)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, shuffle=True)

# print("Training set size:", X_train.shape, Y_train.shape)
# print("Test set size:", X_test.shape, Y_test.shape)