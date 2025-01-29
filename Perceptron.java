import java.util.Arrays;

/**
 * Implements a Rosenblatt Perceptron - a simple binary classifier
 * that can learn linear decision boundaries.
 * Uses threshold θ=0.2 and learning rate α=0.1
 */
public class Perceptron {
    private float[] weights;      // Synaptic weights
    private float threshold = 0.2f;    // θ (theta) - activation threshold
    private float learningRate = 0.01234565657f; // α (alpha) - controls how much weights change

    /**
     * Creates a perceptron with specified number neurons.
     * Initializes weights randomly in range [-1.0, 1.0]
     * @param numInputs Number of input neurons
     */
    public Perceptron(int numInputs) {
        weights = new float[numInputs];
        for (int i = 0; i < numInputs; i++) {
            weights[i] = (float) (Math.random() * 2 - 1); // Random [-1,1]
        }
    }

    /**
     * Calculates perceptron output using step activation function:
     * output = 1 if weighted sum ≥ threshold, 0 otherwise
     * @param inputs Array of input values
     * @return Binary output (0 or 1)
     */
    private float predict(float[] inputs) {
        float sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += inputs[i] * weights[i];  // Calculate weighted sum
        }
        return sum >= threshold ? 1 : 0;    // Step activation function
    }

    /**
     * Trains the perceptron using supervised learning.
     * For each epoch:
     *   1. Calculate output for each training example
     *   2. Calculate error (expected - actual)
     *   3. Update weights using delta rule: w = w + α * error * input
     * 
     * @param data Training input data
     * @param expected Expected outputs
     * @param epochs Number of training iterations
     */
    @Override
    public String toString() {
        return String.format("Perceptron [weights=%s]", 
            Arrays.toString(weights));
    }

    public void train(float[][] data, float[] expected, int maxEpochs) {
        System.out.println(this); // Print initial weights
        
        int epochsNeeded = 0;
        boolean hasError;
        
        do {
            hasError = false;
            for (int i = 0; i < data.length; i++) {
                float output = predict(data[i]);
                float error = expected[i] - output;
                
                if (error != 0) {
                    hasError = true;
                    for (int w = 0; w < weights.length; w++) {
                        weights[w] += learningRate * error * data[i][w];
                    }
                }
            }
            epochsNeeded++;
        } while (hasError && epochsNeeded < maxEpochs);
        
        System.out.printf("Training complete in %d epochs.%n", epochsNeeded);
        System.out.println(this); // Print final weights
    }

    public void test(float[][] data) {
        for (int row = 0; row < data.length; row++) {
            float result = predict(data[row]);
            System.out.printf("Result %d: %.0f%n", row, result);
        }
    }

    public static void main(String[] args) {
        float[][] data = {
            {0.00f, 0.00f},
            {1.00f, 0.00f},
            {0.00f, 1.00f},
            {1.00f, 1.00f}
        };
        float[] expectedAND = {0.00f, 0.00f, 0.00f, 1.00f};
        float[] expectedOR = {0.00f, 1.00f, 1.00f, 1.00f};

        Perceptron p = new Perceptron(2);
        p.train(data, expectedOR, 10000);
        //p.train(data, expectedAND, 10000);
        p.test(data);
    }
}

