import numpy as np

# Activation functions and their derivatives
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def tanh(x): return np.tanh(x)
def tanh_deriv(x): return 1 - x ** 2

# Mapping of activation function names to functions
activation_funcs = {
    'sigmoid': (sigmoid, sigmoid_deriv),
    'relu': (relu, relu_deriv),
    'tanh': (tanh, tanh_deriv)
}

class ANNCustom:
    def __init__(self):
        self.layers = []
        self.activations = []
        self.params = []
        self.cache = {}

    def add_layer(self, input_size, output_size, activation):
        self.layers.append((input_size, output_size))
        self.activations.append(activation)

    def _initialize_params(self):
        np.random.seed(42)
        self.params = [(np.random.randn(input_size, output_size), np.zeros((1, output_size))) 
                       for input_size, output_size in self.layers]

    def _forward(self, X):
        A = X
        self.cache = {'A0': A}
        for i, ((W, b), act) in enumerate(zip(self.params, self.activations)):
            Z = A @ W + b
            A = activation_funcs[act][0](Z)
            self.cache[f'Z{i+1}'], self.cache[f'A{i+1}'] = Z, A
        return self.cache[f'A{len(self.layers)}']

    def _compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _backward(self, y_true):
        grads, L = [], len(self.params)
        dA = self.cache[f'A{L}'] - y_true
        for i in reversed(range(L)):
            Z, A_prev = self.cache[f'Z{i+1}'], self.cache[f'A{i}']
            dZ = dA * activation_funcs[self.activations[i]][1](self.cache[f'A{i+1}'])
            grads.insert(0, (A_prev.T @ dZ / len(y_true), np.sum(dZ, axis=0, keepdims=True) / len(y_true)))
            dA = dZ @ self.params[i][0].T
        return grads

    def _update_params(self, grads, lr):
        self.params = [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(self.params, grads)]

    def compile(self, learning_rate):
        self.learning_rate = learning_rate
        self._initialize_params()

    def fit_once(self, X, y, epochs):
        for epoch in range(epochs):
            y_pred = self._forward(X)
            grads = self._backward(y)
            self._update_params(grads, self.learning_rate)
            self.loss = self._compute_loss(y, y_pred)
        return self
    
#             if (epoch + 1) % 100 == 0:
#                 loss = self._compute_loss(y, y_pred)
#                 print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

    def predict_once(self, X):
        self.pred = 1 if self._forward(X)[0][0]>0.5 else 0
        return self.pred