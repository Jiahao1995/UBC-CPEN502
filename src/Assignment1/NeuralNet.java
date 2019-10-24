package Assignment1;

import Sarb.NeuralNetInterface;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Scanner;

public class NeuralNet implements NeuralNetInterface {

    private final double learningRate;
    private final double momentumTerm;

    private Layer inputLayer;
    private Layer outputLayer;

    private ArrayList<String> errorList = new ArrayList<>();

    /**
     * @param argLearningRate
     * @param argMomentumTerm
     * @param layers is a vararg which must contain one input layer, one output layer, and at least one hidden layer,
     *               otherwise an IllegalArgumentException would be thrown
     */
    public NeuralNet(double argLearningRate, double argMomentumTerm, Layer... layers) {

        learningRate = argLearningRate;
        momentumTerm = argMomentumTerm;

        int N = layers.length;
        if (N < 3)
            throw new IllegalArgumentException("Illegal neural network construction.");
        inputLayer = layers[0];
        outputLayer = layers[N - 1];
        for (int i = 0; i < N; i++) {
            if (i == 0)
                layers[i].connectTo(null, layers[i + 1]);
            else if (i == N - 1)
                layers[i].connectTo(layers[i - 1], null);
            else
                layers[i].connectTo(layers[i - 1], layers[i + 1]);
        }
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
            currLayer.setRandomWeights();
    }

    public void zeroWeights() {
        for (Layer currLayer = inputLayer; currLayer != outputLayer; currLayer = currLayer.getNext())
            currLayer.setZeroWeights();
    }

    public void flip() {
        for (Layer currLayer = inputLayer; currLayer != null; currLayer = currLayer.getNext())
            currLayer.flip();
    }

    public void setTrainingSet(double[] X, double[] y) {
        inputLayer.setInputs(X);
        outputLayer.setOutputs(y);
    }

    public void forwardPropagate() {
        for (Layer currLayer = inputLayer; currLayer != outputLayer; currLayer = currLayer.getNext())
            currLayer.forwardPropagate();
    }

    public void backwardPropagate(double momentumTerm, double learningRate) {
        for (Layer currLayer = outputLayer; currLayer != inputLayer; currLayer = currLayer.getPrev())
            currLayer.backwardPropagate(momentumTerm, learningRate);
    }

    public double outputFor(double[] X) {
        initializeWeights();
        inputLayer.setInputs(X);
        forwardPropagate();
        return outputLayer.getOutputs()[0];
    }

    public double train(double[] X, double argValue) {
        return argValue - outputFor(X);
    }

    public int train(double[][] X, double[][] y) {
        initializeWeights();
        double[] totalError = new double[outputLayer.size()];
        int epoch = 0;
        do {
            for (int i = 0; i < outputLayer.size(); i++)
                totalError[i] = 0;
            for (int i = 0; i < X.length; i++) {
                setTrainingSet(X[i], y[i]);
                forwardPropagate();
                for (int j = 0; j < outputLayer.size(); j++)
                    totalError[j] += Math.pow(y[i][j] - outputLayer.getOutputs()[j], 2);
                backwardPropagate(momentumTerm, learningRate);
            }
            totalError[0] /= 2;
            errorList.add(Double.toString(totalError[0]));
            epoch++;
        } while (totalError[0] > 0.05d);
        return epoch;
    }


    public void save(File argFile) {
        try {
            StringBuilder stringBuilder = new StringBuilder();
            for (Layer currLayer = inputLayer; currLayer != outputLayer; currLayer = currLayer.getNext()) {
                for (int i = 0; i < currLayer.getWeights().length; i++) {
                    for (int j = 0; j < currLayer.getWeights()[0].length; j++)
                        stringBuilder.append(currLayer.getWeights()[i][j] + " ");
                    stringBuilder.append("\n");
                }
            }
            Files.write(argFile.toPath(), stringBuilder.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void load(String argFileName) throws IOException {
        Scanner scanner = new Scanner(new BufferedReader(new FileReader("./weights.txt")));
        for (Layer currLayer = inputLayer; currLayer != outputLayer; currLayer = currLayer.getNext()) {
            for (int i = 0; i < currLayer.getWeights().length; i++) {
                String[] line = scanner.nextLine().trim().split(" ");
                for (int j = 0; j < currLayer.getWeights()[0].length; j++)
                    currLayer.setWeights(i, j, Double.parseDouble(line[j]));
            }
        }
    }

    public void saveError() {
        try {
            Files.write(Paths.get("./error.text"), errorList);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
