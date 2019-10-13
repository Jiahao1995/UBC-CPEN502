package Assignment1;

import Sarb.NeuralNetInterface;

import java.io.File;
import java.io.IOException;

public class NeuralNet implements NeuralNetInterface {

    private final int numInputs;
    private final int numHidden;
    private final int numOutputs = 1;

    private final double learningRate;
    private final double momentumTerm;

    private final double a;
    private final double b;

    private final double errorThreshold = 0.05d;
    private double[] totalError = new double[numOutputs];

    private Layer inputLayer;
    private Layer hiddenLayer;
    private Layer outputLayer;

    public NeuralNet(int argNumInputs, int argNumHidden, double argLearningRate,
                     double argMomentumTerm, double argA, double argB) {
        numInputs = argNumInputs;
        numHidden = argNumHidden;
        learningRate = argLearningRate;
        momentumTerm = argMomentumTerm;
        a = argA;
        b = argB;

        inputLayer = new Layer(numInputs, a, b);
        hiddenLayer = new Layer(numHidden, a, b);
        outputLayer = new Layer(numOutputs, a, b);

        inputLayer.connectTo(null, hiddenLayer);
        hiddenLayer.connectTo(inputLayer, outputLayer);
        outputLayer.connectTo(hiddenLayer, null);
    }

    public double sigmoid(double x) {
        /* defined in Layer */
        return 0.0d;
    }

    public double customSigmoid(double x) {
        /* defined in Layer */
        return 0.0d;
    }

    public void initializeWeights() {
        for (Layer currLayer = inputLayer; currLayer != outputLayer; currLayer = currLayer.getNext())
            currLayer.setRandomWeights(-0.5d, 0.5d);
    }

    public void zeroWeights() {
        for (Layer currLayer = inputLayer; currLayer != outputLayer; currLayer = currLayer.getNext())
            currLayer.setZeroWeights();
    }

    public void forwardPropagate(Layer inputLayer, Layer outputLayer) {
        for (Layer currLayer = inputLayer; currLayer != outputLayer; currLayer = currLayer.getNext())
            currLayer.forwardPropagate();
    }

    public void backwardPropagate(Layer outputLayer, Layer inputLayer,
                                  double momentumTerm, double learningRate) {
        for (Layer currLayer = outputLayer; currLayer != inputLayer; currLayer = currLayer.getPrev())
            currLayer.backwardPropagate(momentumTerm, learningRate);
    }

    public double outputFor(double[] X) {
        initializeWeights();
        inputLayer.setInputs(X);
        forwardPropagate(inputLayer, outputLayer);
        return outputLayer.getOutputs()[0];
    }



    public double train(double[] X, double argValue) {
        return argValue - outputFor(X);
    }

    public int train(double[][] X, double[][] y) {
        initializeWeights();
        int epoch = 0;
        do {
            for (int i = 0; i < numOutputs; i++)
                totalError[i] = 0;
            for (int i = 0; i < X.length; i++) {
                inputLayer.setInputs(X[i]);
                outputLayer.setOutputs(y[i]);
                forwardPropagate(inputLayer, outputLayer);
                for (int j = 0; j < numOutputs; j++)
                    totalError[j] += Math.pow(y[i][j] - outputLayer.getOutputs()[j], 2);
                backwardPropagate(outputLayer, inputLayer, momentumTerm, learningRate);
            }
            totalError[0] /= 2;
            epoch++;
        } while (totalError[0] > errorThreshold);
        System.out.println(epoch);
        return epoch;
    }


    public void save(File argFile) {

    }

    public void load(String argFileName) throws IOException {

    }

    public static void main(String[] args) {
        NeuralNet xor = new NeuralNet(2, 4, 0.2, 0.9, 0, 1);
        xor.train(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
                new double[][]{{0}, {1}, {1}, {0}});
    }

}
