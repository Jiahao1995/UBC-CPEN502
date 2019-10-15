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

    private final double a;
    private final double b;

    public Layer(int numOfNeurons, double argA, double argB) {
        N = numOfNeurons;
        values = new double[N + 1];
        values[N] = 1.d;
        deltas = new double[N];
        a = argA;
        b = argB;
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

    public Layer getPrev() { return prev; }

    public Layer getNext() { return next; }

    public double[][] getWeights() { return currWeights; }

    public void setWeights(int i, int j, double weight) {
        currWeights[i][j] = weight;
    }

    public void setRandomWeights(double lower, double upper) {
        if (next == null)
            return;
        for (int j = 0; j < next.N; j++) {
            for (int i = 0; i < N + 1; i++) {
                Random random = new Random();
                currWeights[i][j] = random.nextDouble() * (upper - lower) + lower;
            }
        }
        prevWeights = currWeights.clone();
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
            throw new NullPointerException("You cannot ask an output layer to forward-propagate!");
        for (int j = 0; j < next.N; j++) {
            next.values[j] = 0.0d;
            for (int i = 0; i < N + 1; i++)
                next.values[j] += currWeights[i][j] * values[i];
            next.values[j] = customSigmoid(next.values[j]);
        }
    }

    public void backwardPropagate(double momentumTerm, double learningRate) {
        if (prev == null)
            throw new NullPointerException("You cannot ask an input layer to backward-propagate!");
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
