package Assignment1;

import Sarb.NeuralNetInterface;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Scanner;

public class NeuralNet implements NeuralNetInterface {

    private final int numInputs;
    private final int numHidden;
    private final int numOutputs = 1;

    private final double learningRate;
    private final double momentumTerm;

    private final double a;
    private final double b;

    private final double lower = -0.5d;
    private final double upper = 0.5d;

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

        inputLayer.flip();
        hiddenLayer.flip();
        outputLayer.flip();

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
            currLayer.setRandomWeights(lower, upper);
    }

    public void zeroWeights() {
        for (Layer currLayer = inputLayer; currLayer != outputLayer; currLayer = currLayer.getNext())
            currLayer.setZeroWeights();
    }

    public void flip() {
        for (Layer currLayer = inputLayer; currLayer != null; currLayer = currLayer.getNext())
            currLayer.flip();
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
        int epoch = 0;
        do {
            for (int i = 0; i < numOutputs; i++)
                totalError[i] = 0;
            for (int i = 0; i < X.length; i++) {
                inputLayer.setInputs(X[i]);
                outputLayer.setOutputs(y[i]);
                forwardPropagate();
                for (int j = 0; j < numOutputs; j++)
                    totalError[j] += Math.pow(y[i][j] - outputLayer.getOutputs()[j], 2);
                backwardPropagate(momentumTerm, learningRate);
            }
            totalError[0] /= 2;
//            System.out.println(totalError[0]);
            epoch++;
        } while (totalError[0] > errorThreshold);
//        System.out.println(epoch);
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


}
