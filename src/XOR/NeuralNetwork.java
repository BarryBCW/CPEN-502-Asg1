package XOR;

import java.text.*;
import java.util.*;
import java.io.*;
import java.lang.Math;
public class NeuralNetwork {
    static {
        Locale.setDefault(Locale.ENGLISH);
    }
    
    final DecimalFormat df;
    final Random rand = new Random();
    final ArrayList<Neuron> inputLayer = new ArrayList<Neuron>();
    final ArrayList<Neuron> hiddenLayer = new ArrayList<Neuron>();
    final ArrayList<Neuron> outputLayer = new ArrayList<Neuron>();
    final Neuron bias = new Neuron();
    final int[] layers;
 
    final double epsilon = 0.00000000001;
 
    final double learningRate = 0.2;
    final double momentum = 0.0f;
 
    // Inputs for xor problem
    //final double inputs[][] = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } };
    final double inputs[][] = new double[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    // Corresponding outputs, xor training data
    final double expectedOutputs[][] =  new double[][]{{-1}, {1}, {1}, {-1}};
    //final double expectedOutputs[][] = { { 0 }, { 1 }, { 1 }, { 0 } };
    //double resultOutputs[][] = { { -1 }, { -1 }, { -1 }, { -1 } }; 
    double resultOutputs[][] = { { 0 }, { 0 }, { 0 }, { 0 } }; 
    double output[];
 
    // for weight update all
    final HashMap<String, Double> weightUpdate = new HashMap<String, Double>();
 
    public static void main(String[] args) throws IOException {
        NeuralNetwork nn = new NeuralNetwork(2, 4, 1);
        int maxRuns = 50000;
        double minErrorCondition = 0.05;
        nn.run(maxRuns, minErrorCondition);
    }
 
    public NeuralNetwork(int input, int hidden, int output) {
        this.layers = new int[] { input, hidden, output };
        df = new DecimalFormat("#.0#");
 
        /**
         * Create all neurons and connections Connections are created in the
         * neuron class
         */
        for (int i = 0; i < layers.length; i++) {
            if (i == 0) { // input layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    inputLayer.add(neuron);
                }
            } else if (i == 1) { // hidden layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addInConnectionsS(inputLayer);
                    neuron.addBiasConnection(bias);
                    hiddenLayer.add(neuron);
                }
            }
 
            else if (i == 2) { // output layer
                for (int j = 0; j < layers[i]; j++) {
                    Neuron neuron = new Neuron();
                    neuron.addInConnectionsS(hiddenLayer);
                    neuron.addBiasConnection(bias);
                    outputLayer.add(neuron);
                }
            } else {
                System.out.println("Error NeuralNetwork init");
            }
        }
 
        // initialize random weights
        for (Neuron neuron : hiddenLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            
            }
        }
        for (Neuron neuron : outputLayer) {
            ArrayList<Connection> connections = neuron.getAllInConnections();
            for (Connection conn : connections) {
                double newWeight = getRandom();
                conn.setWeight(newWeight);
            }
        }
 
        // reset id counters
        Neuron.counter = 0;
        Connection.counter = 0;
 
       
    }
 
    // random
    double getRandom() {
    	 
        return (Math.random() - 0.5); // [-0.5;0.5]
    }
 
    /**
     * 
     * @param inputs
     *            There is equally many neurons in the input layer as there are
     *            in input variables
     */
    public void setInput(double inputs[]) {
        for (int i = 0; i < inputLayer.size(); i++) {
            inputLayer.get(i).setOutput(inputs[i]);
        }
    }
 
    public double[] getOutput() {
        double[] outputs = new double[outputLayer.size()];
        for (int i = 0; i < outputLayer.size(); i++)
            outputs[i] = outputLayer.get(i).getOutput();
        return outputs;
    }
 
    /**
     * Calculate the output of the neural network based on the input The forward
     * operation
     */
    public void activate() {
        for (Neuron n : hiddenLayer)
            n.calculateOutput();
        for (Neuron n : outputLayer)
            n.calculateOutput();
    }
 
    /**
     * all output propagate back
     * 
     * @param expectedOutput
     *            first calculate the partial derivative of the error with
     *            respect to each of the weight leading into the output neurons
     *            bias is also updated here
     */
    public void applyBackpropagation(double expectedOutput[]) {
 
        /* error check, normalize value ]0;1[
        for (int i = 0; i < expectedOutput.length; i++) {
            double d = expectedOutput[i];
            if (d < -1 || d > 1) {
                if (d < 0)
                    expectedOutput[i] = -1 + epsilon;
                else
                    expectedOutput[i] = 1 - epsilon;
            }
        } */
 
        int i = 0;
        for (Neuron n : outputLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            double ak = n.getOutput();
            double desiredOutput = expectedOutput[i];
          //double partialDerivative = ak * (1 - ak) * (desiredOutput - ak);         
            double partialDerivative = 0.5 * (1 - Math.pow(ak, 2)) * (desiredOutput - ak);
            for (Connection con : connections) {            
                double ai = con.leftNeuron.getOutput();                                 
                double deltaWeight = momentum * con.getPrevDeltaWeight()+learningRate * partialDerivative* ai;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight);
            }
            i++;
        }
 
        // update weights for the hidden layer
        for (Neuron n : hiddenLayer) {
            ArrayList<Connection> connections = n.getAllInConnections();
            double aj = n.getOutput();
            double sumKoutputs = 0;
            int j = 0;
            for (Neuron out_neu : outputLayer) {
                double wjk = out_neu.getConnection(n.id).getWeight();
                double desiredOutput = (double) expectedOutput[j];
                double ak = out_neu.getOutput();
                j++;
                sumKoutputs += 0.5 * (1 - Math.pow(ak, 2)) * (desiredOutput - ak)*wjk;
                        //+ ((desiredOutput - ak) * ak * (1 - ak) * wjk);
            }
          //double partialDerivative = aj * (1 - aj) * sumKoutputs;
            double partialDerivative = 0.5* (1 - Math.pow(aj, 2)) * sumKoutputs;
            for (Connection con : connections) {                
                double ai = con.leftNeuron.getOutput();               
                double deltaWeight = momentum * con.getPrevDeltaWeight()+learningRate * partialDerivative* ai;
                double newWeight = con.getWeight() + deltaWeight;
                con.setDeltaWeight(deltaWeight);
                con.setWeight(newWeight);
            }
        }
    }
 
    void run(int maxSteps, double minError)throws IOException {
        int i;
        // Train neural network until minError reached or maxSteps exceeded
        double error = 1;
        double [] ErrorList = new double[50000];
        
        for (i = 0; i < maxSteps && error > minError; i++) {
            error = 0;
            for (int p = 0; p < inputs.length; p++) {
                setInput(inputs[p]);
 
                activate();
 
                output = getOutput();
                resultOutputs[p] = output;       
              
                double err = Math.pow(output[0] - expectedOutputs[p][0], 2);
                error += err;
                
                applyBackpropagation(expectedOutputs[p]);
                
            }
            error = error/2;
            ErrorList[i] = error;
            
        }
        printResult();
        File file = new File("chart.txt");
		FileWriter out = new FileWriter(file);
		for(int q = 0; q<i; q++) {
			out.write(ErrorList[q]+"\t");
			out.write("\r\n");
		}
		out.close();
        System.out.println(ErrorList);
        System.out.println("Sum of squared errors = " + error);
        System.out.println("##### EPOCH " + i+"\n");
        if (i == maxSteps) {
            System.out.println("!Error training try again");
        } 
    }
     
    void printResult()
    {
        System.out.println("NN example with xor training");
        for (int p = 0; p < inputs.length; p++) {
            System.out.print("INPUTS: ");
            for (int x = 0; x < layers[0]; x++) {
                System.out.print(inputs[p][x] + " ");
            }
 
            System.out.print("EXPECTED: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.print(expectedOutputs[p][x] + " ");
            }
 
            System.out.print("ACTUAL: ");
            for (int x = 0; x < layers[2]; x++) {
                System.out.print(resultOutputs[p][x] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }
 
    String weightKey(int neuronId, int conId) {
        return "N" + neuronId + "_C" + conId;
    }
 
    
    
}