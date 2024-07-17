import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define domain dimensions
length = 2.2
height = 0.41
cylinder_center = (0.2, 0.2)
cylinder_radius = 0.05

# Generate a grid of points within the domain
x = np.linspace(0, length, 100)
y = np.linspace(0, height, 50)
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()

# Remove points inside the cylinder
points = np.array([X, Y]).T
mask = np.sqrt((points[:, 0] - cylinder_center[0])**2 + (points[:, 1] - cylinder_center[1])**2) > cylinder_radius
X = points[mask, 0]
Y = points[mask, 1]

# Define initial conditions for velocity and pressure
def initial_velocity(x, y):
    return 4.0 * 1.5 * y * (0.41 - y) / 0.41**2, 0

def initial_pressure(x, y):
    return 0

# Generate synthetic data
U = np.array([initial_velocity(x, y) for x, y in zip(X, Y)])
P = np.array([initial_pressure(x, y) for x, y in zip(X, Y)])

# Stack coordinates, velocity, and pressure for training
training_data = np.hstack((X[:, None], Y[:, None], U, P[:, None]))

# Separate inputs and outputs
X_train = training_data[:, :2]
U_train = training_data[:, 2:4]
P_train = training_data[:, 4:]




# Define neural network model
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense4 = tf.keras.layers.Dense(3)  # Output: u, v, p

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

# Define the loss function
def loss_function(model, X_train, U_train, P_train):
    u_v_p_pred = model(X_train)
    u_pred, v_pred, p_pred = u_v_p_pred[:, 0], u_v_p_pred[:, 1], u_v_p_pred[:, 2]
    
    # Calculate residuals
    residual_u = U_train[:, 0] - u_pred
    residual_v = U_train[:, 1] - v_pred
    residual_p = P_train[:, 0] - p_pred
    
    loss_u = tf.reduce_mean(tf.square(residual_u))
    loss_v = tf.reduce_mean(tf.square(residual_v))
    loss_p = tf.reduce_mean(tf.square(residual_p))
    
    return loss_u + loss_v + loss_p

# Training the PINN
def train_pinn(model, X_train, U_train, P_train, epochs=1000, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = loss_function(model, X_train, U_train, P_train)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Create and train the model
pinn_model = PINN()
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
U_train_tf = tf.convert_to_tensor(U_train, dtype=tf.float32)
P_train_tf = tf.convert_to_tensor(P_train, dtype=tf.float32)

train_pinn(pinn_model, X_train_tf, U_train_tf, P_train_tf)




# Evaluate the model at t=3 seconds
X_eval = np.vstack((X_train, 3 * np.ones(X_train.shape[0])))  # Add time as a third dimension if needed
X_eval_tf = tf.convert_to_tensor(X_eval, dtype=tf.float32)
u_v_p_eval = pinn_model(X_eval_tf)
u_eval, v_eval, p_eval = u_v_p_eval[:, 0], u_v_p_eval[:, 1], u_v_p_eval[:, 2]

# Plot the results
plt.figure(figsize=(10, 5))
plt.quiver(X_train[:, 0], X_train[:, 1], u_eval, v_eval, scale=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity Field at t=3s')
plt.show()

