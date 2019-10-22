package Assignment1;

public class OneLayerTest {

    public static void binaryTest() {

        double[][] X = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        double[][] y = { {0}, {1}, {1}, {0} };

        Layer inputLayer = new Layer(2);
        Layer hiddenLayer = new Layer(4);
        Layer outputLayer = new Layer(1);
        NeuralNet xor = new NeuralNet(0.2, 0, inputLayer, hiddenLayer, outputLayer);

        System.out.println(xor.train(X, y));
        xor.saveError();
    }

    public static void bipolarTest() {

        double[][] X = { {-1, -1}, {-1, 1}, {1, -1}, {1, 1} };
        double[][] y = { {-1}, {1}, {1}, {-1} };

        Layer inputLayer = new Layer(2);
        Layer hiddenLayer = new Layer(4);
        Layer outputLayer = new Layer(1);
        NeuralNet xor = new NeuralNet(0.2, 0, inputLayer, hiddenLayer, outputLayer);
        xor.flip();

        System.out.println(xor.train(X, y));
        xor.saveError();
    }

    public static void main(String[] args) {
        bipolarTest();
    }

}
