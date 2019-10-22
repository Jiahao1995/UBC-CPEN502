# Code

file:///Users/lijiahao/Documents/UBC/CPEN_502/UBC-CPEN502/src/Assignment1/Layer.java

```java
package Assignment1;

import java.util.Random;

public class Layer {

    private static boolean BINARY = true;

    private Layer prev;
    private Layer next;
    private final int N;
    private double[] values;
    private double[] deltas;
    private double[][] currWeights;
    private double[][] prevWeights;
    private double[] requiredOutputs;

    private final double a = 0.0d;
    private final double b = 1.0d;

    public Layer(int numOfNeurons) {
        N = numOfNeurons;
        values = new double[N + 1];
        values[N] = 1.d;
        deltas = new double[N];
    }

    public void connectTo(Layer prevLayer, Layer nextLayer) {
        prev = prevLayer;
        next = nextLayer;
        if (next != null) {
            currWeights = new double[N + 1][next.N];
            prevWeights = new double[N + 1][next.N];
            requiredOutputs = null;
        } else {
            requiredOutputs = new double[N];
            currWeights = null;
            prevWeights = null;
        }
    }

    public void setInputs(double[] inputs) {
        if (prev != null)
            return;
        if (inputs.length != N)
            throw new IllegalArgumentException();
        for (int i = 0; i < inputs.length; i++)
            values[i] = inputs[i];
    }

    public void setOutputs(double[] outputs) {
        if (next != null)
            return;
        if (outputs.length != N)
            throw new IllegalArgumentException();
        for (int i = 0; i < outputs.length; i++)
            requiredOutputs[i] = outputs[i];
    }

    public double[] getOutputs() { return values; }

    public int size() { return N; }

    public Layer getPrev() { return prev; }

    public Layer getNext() { return next; }

    public double[][] getWeights() { return currWeights; }

    public void setWeights(int i, int j, double weight) {
        currWeights[i][j] = weight;
    }

    public void setRandomWeights() {
        double lower = -0.5d;
        double upper = 0.5d;
        if (next == null)
            return;
        for (int j = 0; j < next.N; j++) {
            for (int i = 0; i < N + 1; i++) {
                Random random = new Random();
                currWeights[i][j] = random.nextDouble() * (upper - lower) + lower;
                prevWeights[i][j] = currWeights[i][j];
            }
        }
    }

    public void setZeroWeights() {
        if (next == null)
            return;
        for (int j = 0; j < next.N; j++) {
            for (int i = 0; i < N + 1; i++)
                currWeights[i][j] = 0.0d;
        }
        prevWeights = currWeights.clone();
    }

    public double sigmoid(double x) {
        return 2 / (1 + Math.exp(-x)) - 1;
    }

    public double customSigmoid(double x) {
        if (BINARY)
            return (b - a) / (1 + Math.exp(-x)) + a;
        else
            return sigmoid(x);
    }

    public void forwardPropagate() {
        if (next == null)
            throw new NullPointerException("Cannot perform a forward-propagation on an output layer.");
        for (int j = 0; j < next.N; j++) {
            next.values[j] = 0.0d;
            for (int i = 0; i < N + 1; i++)
                next.values[j] += currWeights[i][j] * values[i];
            next.values[j] = customSigmoid(next.values[j]);
        }
    }

    public void backwardPropagate(double momentumTerm, double learningRate) {
        if (prev == null)
            throw new NullPointerException("Cannot perform a backward-propagation on an input layer.");
        if (next == null) {
            for (int i = 0; i < N; i++) {
                deltas[i] = 0.0d;
                double error = requiredOutputs[i] - values[i];
                double derivative;
                if (BINARY)
                    derivative = values[i] * (1 - values[i]);
                else
                    derivative = (values[i] + 1) * 0.5 * (1 - values[i]);
                deltas[i] = derivative * error;
            }
        } else {
            for (int i = 0; i < N; i++) {
                deltas[i] = 0.0d;
                double derivative;
                if (BINARY)
                    derivative = values[i] * (1 - values[i]);
                else
                    derivative = (values[i] + 1) * 0.5 * (1 - values[i]);
                for (int j = 0; j < next.N; j++)
                    deltas[i] += currWeights[i][j] * next.deltas[j];
                deltas[i] = deltas[i] * derivative;
            }
        }
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < prev.N + 1; i++) {
                double deltaWeight = prev.currWeights[i][j] - prev.prevWeights[i][j];
                prev.prevWeights[i][j] = prev.currWeights[i][j];
                prev.currWeights[i][j] += momentumTerm * deltaWeight + learningRate * deltas[j] * prev.values[i];
            }
        }
    }

    public void flip() {
        BINARY = !BINARY;
    }

}
```

file:///Users/lijiahao/Documents/UBC/CPEN_502/UBC-CPEN502/src/Assignment1/NeuralNet.java

```java
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
```

#Result

![binary00](/Users/lijiahao/Documents/UBC/CPEN_502/UBC-CPEN502/binary00.png)

![binary09](/Users/lijiahao/Documents/UBC/CPEN_502/UBC-CPEN502/binary09.png)

![bipolar00](/Users/lijiahao/Documents/UBC/CPEN_502/UBC-CPEN502/bipolar00.png)

![bipolar09](/Users/lijiahao/Documents/UBC/CPEN_502/UBC-CPEN502/bipolar09.png)