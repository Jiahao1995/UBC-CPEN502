package Assignment1;

public class Main {

    public static void main(String[] args) {

        NeuralNet xor = new NeuralNet(2, 4, 0.2, 0.9, 0, 1);
        int epoch = 0;

//        for (int i = 0; i < 100; i++) {
//            epoch += xor.train(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}}, new double[][]{{0}, {1}, {1}, {0}});
//        }

        for (int i = 0; i < 100; i++) {
            epoch += xor.train(new double[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}},
                    new double[][]{{-1}, {1}, {1}, {-1}});
        }

        System.out.println(epoch / 100);

    }

}
