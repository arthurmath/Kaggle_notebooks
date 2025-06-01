import numpy as np
import tensorflow as tf
print("tf imported")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, Reshape, Concatenate, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
print("imports finished")


# Paramètres constants
G = 9.81
AIR_DENSITY = 1.225

# Fonction de simulation (version corrigée)
def rocket_simulation(rocket_area, initial_mass, fuel_mass, burn_rate, thrust, drag_coefficient):
    """ Simulates a 1D rocket launch with improved physics """
    altitude = 0.1
    velocity = 0.0
    mass = initial_mass
    dt = 1
    t_max = 300
    
    altitudes = []
    times = np.arange(0, t_max, dt)
    
    for time in times:
        # Fuel consumption
        if fuel_mass > 0:
            dm = min(burn_rate * dt, fuel_mass)
            fuel_mass -= dm
            mass -= dm
            current_thrust = thrust
        else:
            current_thrust = 0

        # Physics calculations
        weight = mass * G
        drag = 0.5 * rocket_area * drag_coefficient * AIR_DENSITY * abs(velocity) * velocity
        net_force = current_thrust - weight - drag
        acceleration = net_force / mass
        
        # Integration (Semi-implicit Euler)
        velocity += acceleration * dt
        altitude += velocity * dt
        
        altitudes.append(max(altitude, 0))
        
    return np.array(altitudes)

# Génération des données synthétiques
def generate_dataset(n_samples):
    X = np.zeros((n_samples, 6))
    y = np.zeros((n_samples, 300))
    
    # Plages réalistes des paramètres
    param_ranges = [
        (0.1, 10.0),      # rocket_area (m²)
        (1000, 100000),    # initial_mass (kg)
        (500, 50000),      # fuel_mass (kg)
        (10, 1000),       # burn_rate (kg/s)
        (10000, 5000000), # thrust (N)
        (0.1, 1.0)        # drag_coefficient
    ]
    
    for i in range(n_samples):
        params = [np.random.uniform(low, high) for low, high in param_ranges]
        X[i] = params
        y[i] = rocket_simulation(*params)
    
    return X, y

# Génération des données
print("Génération des données...")
X, y = generate_dataset(10000)  # Adaptez selon vos ressources

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
param_scaler = StandardScaler()
X_train_scaled = param_scaler.fit_transform(X_train)
X_test_scaled = param_scaler.transform(X_test)

output_scaler = StandardScaler()
y_train_scaled = output_scaler.fit_transform(y_train)
y_test_scaled = output_scaler.transform(y_test)

# Séquence de temps (identique pour toutes les simulations)
time_steps = np.tile(np.arange(300), (X_train_scaled.shape[0], 1))
time_steps_test = np.tile(np.arange(300), (X_test_scaled.shape[0], 1))

# Normalisation du temps
time_scaler = StandardScaler()
time_steps_scaled = time_scaler.fit_transform(time_steps)
time_steps_test_scaled = time_scaler.transform(time_steps_test)

# Architecture du modèle
def create_model():
    # Entrée 1: Paramètres de la fusée
    input_params = Input(shape=(6,))
    encoded = Dense(256, activation='relu')(input_params)
    repeated = RepeatVector(300)(encoded)
    
    # Entrée 2: Séquence temporelle
    input_times = Input(shape=(300,))
    times_reshaped = Reshape((300, 1))(input_times)
    
    # Concaténation des caractéristiques
    concat = Concatenate(axis=-1)([repeated, times_reshaped])
    
    # Décodeur LSTM
    x = LSTM(256, return_sequences=True, activation='tanh')(concat)
    outputs = TimeDistributed(Dense(1))(x)
    
    model = Model(inputs=[input_params, input_times], outputs=outputs)
    return model

# Création et compilation du modèle
model = create_model()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Entraînement
print("Début de l'entraînement...")
history = model.fit(
    [X_train_scaled, time_steps_scaled], 
    y_train_scaled.reshape(-1, 300, 1),
    validation_split=0.15,
    epochs=50,
    batch_size=64,
    verbose=1
)

# Évaluation sur le test set
print("Évaluation sur le jeu de test...")
test_loss = model.evaluate(
    [X_test_scaled, time_steps_test_scaled],
    y_test_scaled.reshape(-1, 300, 1),
    verbose=0
)
print(f"RMSE: {np.sqrt(test_loss[0]):.4f}")

# Prédictions sur le jeu de test
y_pred_scaled = model.predict([X_test_scaled, time_steps_test_scaled])
y_pred = output_scaler.inverse_transform(y_pred_scaled.reshape(-1, 300))

# Visualisation des résultats
def plot_sample(i):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[i], label='Vérité terrain')
    plt.plot(y_pred[i], '--', label='Prédiction')
    plt.title(f"Courbe d'altitude - Échantillon {i}")
    plt.xlabel('Temps (s)')
    plt.ylabel('Altitude (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Exemple de visualisation
plot_sample(0)
plot_sample(42)