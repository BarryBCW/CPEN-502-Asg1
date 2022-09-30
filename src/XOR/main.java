package XOR;

import java.io.IOException;

public class main {
	public static void main(String[] args) throws IOException  {
        NeuralNetwork nn = new NeuralNetwork(0.2,0.9);
        int maxRuns = 50000;
        double minErrorCondition = 0.05;
        nn.setWeights();
        nn.run(maxRuns, minErrorCondition);
    }
}
