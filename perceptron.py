import numpy as np

class Perceptron:
    def __init__(self, num_inputs):
        """Initialize perceptron with random weights in [-1,1]"""
        self.weights = np.random.uniform(-1.0, 1.0, num_inputs)
        self.threshold = 0.2  # θ (theta)
        self.learning_rate = 0.1  # α (alpha)

    def predict(self, inputs):
        """Calculate output using step activation function"""
        weighted_sum = np.dot(inputs, self.weights)
        return 1.0 if weighted_sum >= self.threshold else 0.0

    def train(self, data, expected, epochs):
        """Train perceptron using delta rule"""
        for _ in range(epochs):
            for inputs, target in zip(data, expected):
                output = self.predict(inputs)
                error = target - output
                # Update weights: Δw = α * error * input
                self.weights += self.learning_rate * error * inputs

if __name__ == "__main__":
    # Training data for AND gate
    data = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])
    expectedAND = np.array([0.0, 0.0, 0.0, 1.0])
    expectedOR = np.array([0.0, 1.0, 1.0, 1.0])
    # Create and train perceptron
    p = Perceptron(2)
    p.train(data, expectedAND, 10000)
    #p.train(data, expectedOR, 10000)
    # Test results
    for inputs in data:
        print(f"Input: {inputs}, Output: {p.predict(inputs)}")