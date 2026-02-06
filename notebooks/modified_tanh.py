class ScaledTanhNeuron:
    def __init__(self, input_dim):
        # Initialize weights randomly and bias to zero
        self.w = np.random.randn(input_dim, 1) * 0.01
        self.b = 0.0

    def forward(self, X):
        """
        Forward propagation with Scaled Tanh.
        Range: [-1, 1] -> Scaled to -> [0, 1]
        """
        self.X_input = X
        self.Z = np.dot(X, self.w) + self.b
        
        # Raw Tanh output (-1 to 1)
        self.raw_tanh = np.tanh(self.Z)
        
        # Scale it to (0 to 1)
        self.A = 0.5 * (self.raw_tanh + 1)
        
        return self.A

    def backward(self, Y, learning_rate):
        m = Y.shape[0]
        
        # 1. Gradient of Loss with respect to A (BCE derivative)
        epsilon = 1e-15
        A_clipped = np.clip(self.A, epsilon, 1 - epsilon)
        dA = - (np.divide(Y, A_clipped) - np.divide(1 - Y, 1 - A_clipped))
        
        # 2. Gradient of A with respect to Z
        # The derivative of 0.5 * (tanh(Z) + 1) is: 0.5 * (1 - tanh^2(Z))
        tanh_grad = 1 - self.raw_tanh**2
        dZ = dA * 0.5 * tanh_grad  # Notice the 0.5 scaling factor here
        
        # 3. Gradients for weights and bias
        dw = (1 / m) * np.dot(self.X_input.T, dZ)
        db = (1 / m) * np.sum(dZ)
        
        # 4. Update
        self.w -= learning_rate * dw
        self.b -= learning_rate * db

    def compute_loss(self, Y):
        m = Y.shape[0]
        epsilon = 1e-15
        A_clipped = np.clip(self.A, epsilon, 1 - epsilon)
        return - (1/m) * np.sum(Y * np.log(A_clipped) + (1 - Y) * np.log(1 - A_clipped))

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# --- Test the new Scaled Tanh Neuron ---
# (Assuming X and Y are already generated from the previous code)

neuron_scaled_tanh = ScaledTanhNeuron(input_dim=X.shape[1])
loss_history = []

print("--- Training Scaled Tanh ---")
for i in range(1000):
    neuron_scaled_tanh.forward(X)
    loss = neuron_scaled_tanh.compute_loss(Y)
    loss_history.append(loss)
    neuron_scaled_tanh.backward(Y, learning_rate=0.1)
    
    if i % 200 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

# Visualize the result
plt.figure(figsize=(6, 5))
plt.plot(loss_history)
plt.title("Scaled Tanh Loss Curve (Smooth Convergence)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()