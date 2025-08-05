import numpy as np
import matplotlib.pyplot as plt

# Define inputs and true outputs (supervised learning)
inputs = np.array([[1, 1], [2, 3], [3, 5], [4, 7]])  # Input features
true_outputs = np.array([[0.5, 1], [1, 1.5], [1.5, 2.5], [2, 3]])  # Ground truth joint angles

# Initialize weights and biases randomly
weights = np.random.rand(2, 2)  # 2 inputs, 2 neurons
biases = np.random.rand(2)  # 2 neurons

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# To store error over epochs for visualization
errors_over_epochs = []

# Training loop
for epoch in range(epochs):
    total_error = 0

    for i in range(len(inputs)):
        # Forward pass
        input_sample = inputs[i]  # Single input sample
        true_output = true_outputs[i]  # Corresponding true output

        # Compute weighted sum and apply activation function
        z = np.dot(input_sample, weights) + biases  # Weighted sum
        output = sigmoid(z)  # Activated output (prediction)

        # Compute the error (mean squared error for each output)
        error = (true_output - output) ** 2
        total_error += np.sum(error)

        # Backward pass (compute gradients)
        output_error = 2 * (output - true_output)  # Derivative of MSE w.r.t. output
        output_delta = output_error * sigmoid_derivative(z)  # Delta for the layer

        # Gradient of weights and biases
        weight_gradients = np.outer(input_sample, output_delta)  # Input * delta
        bias_gradients = output_delta  # Delta directly affects biases

        # Update weights and biases
        weights -= learning_rate * weight_gradients
        biases -= learning_rate * bias_gradients

    # Store total error for this epoch
    errors_over_epochs.append(total_error)

    # Print error every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total Error: {total_error:.4f}")

# Plot the error over epochs
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), errors_over_epochs, label="Total Error", color="blue")
plt.title("Error Over Epochs", fontsize=16)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Total Error", fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.show()

# Visualize predictions vs true values
print("\nFinal Weights:")
print(weights)
print("\nTesting on new data:")
test_inputs = inputs  # Testing on training data for simplicity
predicted_outputs = []

for test_input in test_inputs:
    z = np.dot(test_input, weights) + biases
    output = sigmoid(z)
    predicted_outputs.append(output)

predicted_outputs = np.array(predicted_outputs)

# Plot predictions vs true values
plt.figure(figsize=(8, 6))
plt.plot(range(len(true_outputs)), true_outputs[:, 0], label="True Joint Angle 1", linestyle="--", marker="o", color="green")
plt.plot(range(len(predicted_outputs)), predicted_outputs[:, 0], label="Predicted Joint Angle 1", linestyle="-", marker="o", color="blue")
plt.plot(range(len(true_outputs)), true_outputs[:, 1], label="True Joint Angle 2", linestyle="--", marker="x", color="orange")
plt.plot(range(len(predicted_outputs)), predicted_outputs[:, 1], label="Predicted Joint Angle 2", linestyle="-", marker="x", color="red")
plt.title("True vs Predicted Joint Angles", fontsize=16)
plt.xlabel("Data Point Index", fontsize=12)
plt.ylabel("Joint Angles", fontsize=12)
plt.legend(fontsize=12)
plt.grid()
plt.show()
