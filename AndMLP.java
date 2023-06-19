import java.util.Arrays;

public class AndMLP {
    private double[][] weights1;
    private double[][] weights2;
    private double learningRate;
    private int hiddenNeurons;
    private int epochs;


    public AndMLP(int length, int hiddenNeurons2, int length2, double learningRate2, int epochs2) {
        this.len
    }


    public static void main(String[] args) {
        // Dados de treinamento para a porta lógica AND
        double[][] X_train = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        double[][] y_train = {{1}, {-1}, {-1}, {-1}};

        // Definição dos parâmetros da MLP
        int hiddenNeurons = 2;
        double learningRate = 0.1;
        int epochs = 10000;

        // Criação e treinamento da MLP
        AndMLP mlp = new AndMLP(X_train[0].length, hiddenNeurons, y_train[0].length, learningRate, epochs);
        mlp.train(X_train, y_train);

        // Dados de teste para a porta lógica AND
        double[][] X_test = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

        // Realiza as previsões
        double[] predictions = mlp.predict(X_test);

        // Resultados
        for (int i = 0; i < X_test.length; i++) {
            System.out.printf("Entrada: %s, Saída esperada: %.3f, Saída obtida: %.3f%n",
                    Arrays.toString(X_test[i]), y_train[i][0], predictions[i]);
        }
    }


    public MLP(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate, int epochs) {
        this.weights1 = new double[inputNeurons][hiddenNeurons];
        this.weights2 = new double[hiddenNeurons][outputNeurons];
        this.learningRate = learningRate;
        this.hiddenNeurons = hiddenNeurons;
        this.epochs = epochs;

        initializeWeights();
    }

    private void initializeWeights() {
        for (double[] weights : weights1) {
            Arrays.fill(weights, Math.random());
        }

        for (double[] weights : weights2) {
            Arrays.fill(weights, Math.random());
        }
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    public void train(double[][] X, double[][] y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < X.length; i++) {
                // Forward pass
                double[] layer1Output = new double[hiddenNeurons];
                double[] layer2Output = new double[y[i].length];

                for (int j = 0; j < hiddenNeurons; j++) {
                    double sum = 0;

                    for (int k = 0; k < X[i].length; k++) {
                        sum += X[i][k] * weights1[k][j];
                    }

                    layer1Output[j] = sigmoid(sum);
                }

                for (int j = 0; j < y[i].length; j++) {
                    double sum = 0;

                    for (int k = 0; k < layer1Output.length; k++) {
                        sum += layer1Output[k] * weights2[k][j];
                    }

                    layer2Output[j] = sigmoid(sum);
                }

                // Backpropagation
                double[] layer2Error = new double[y[i].length];
                double[] layer2Gradient = new double[y[i].length];

                for (int j = 0; j < y[i].length; j++) {
                    layer2Error[j] = y[i][j] - layer2Output[j];
                    layer2Gradient[j] = layer2Error[j] * sigmoidDerivative(layer2Output[j]);
                }

                double[] layer1Error = new double[hiddenNeurons];
                double[] layer1Gradient = new double[hiddenNeurons];

                for (int j = 0; j < hiddenNeurons; j++) {
                    double sum = 0;

                    for (int k = 0; k < y[i].length; k++) {
                        sum += layer2Gradient[k] * weights2[j][k];
                    }

                    layer1Error[j] = sum;
                    layer1Gradient[j] = layer1Error[j] * sigmoidDerivative(layer1Output[j]);
                }

                // Update weights
                for (int j = 0; j < X[i].length; j++) {
                    for (int k = 0; k < hiddenNeurons; k++) {
                        weights1[j][k] += learningRate * X[i][j] * layer1Gradient[k];
                    }
                }

                for (int j = 0; j < hiddenNeurons; j++) {
                    for (int k = 0; k < y[i].length; k++) {
                        weights2[j][k] += learningRate * layer1Output[j] * layer2Gradient[k];
                    }
                }
            }
        }
    }

    public double[] predict(double[][] X) {
        double[] predictions = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            double[] layer1Output = new double[hiddenNeurons];
            double[] layer2Output = new double[weights2[0].length];

            for (int j = 0; j < hiddenNeurons; j++) {
                double sum = 0;

                for (int k = 0; k < X[i].length; k++) {
                    sum += X[i][k] * weights1[k][j];
                }

                layer1Output[j] = sigmoid(sum);
            }

            for (int j = 0; j < weights2[0].length; j++) {
                double sum = 0;

                for (int k = 0; k < layer1Output.length; k++) {
                    sum += layer1Output[k] * weights2[k][j];
                }

                layer2Output[j] = sigmoid(sum);
            }

            predictions[i] = layer2Output[0]; // Assuming only one output neuron
        }

        return predictions;
    }
}
