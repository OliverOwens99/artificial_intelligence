

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class EncogRunner {

    public static void main(String[] args) {
        // Load dataset from GameRunner class
        double[][] data = GameRunner.data;
        double[][] expected = GameRunner.expected;
        MLDataSet trainingSet = new BasicMLDataSet(data, expected);

        // Create a basic neural network
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, 4)); // Input layer with 4 nodes
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 2)); // Hidden layer with 2 nodes
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 4)); // Output layer with 4 nodes
        network.getStructure().finalizeStructure();
        network.reset();

        // Train the neural network
        MLTrain train = new ResilientPropagation(network, trainingSet);
        double minError = 0.28; // Change and see the effect on the result
        int epoch = 1;
        do {
            train.iteration();
            System.out.println("Epoch #" + epoch + " Error: " + train.getError());
            epoch++;
        } while (train.getError() > minError);
        train.finishTraining();

        // Test the neural network
        for (MLDataPair pair : trainingSet) {
            MLData output = network.compute(pair.getInput());
            System.out.println(pair.getInput().getData(0) + ","
                    + pair.getInput().getData(1)
                    + ", Y=" + (int) Math.round(output.getData(0)) // Round the result
                    + ", Yd=" + (int) pair.getIdeal().getData(0));
        }

        // Shutdown the neural network
        Encog.getInstance().shutdown();
    }
}