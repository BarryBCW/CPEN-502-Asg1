package XOR;


import java.util.*;
import java.io.*;
import java.lang.Math;
public class NeuralNetwork{
    static {
        Locale.setDefault(Locale.ENGLISH);
    }
    
    
    final Random rand = new Random();
    
    final double epsilon = 0.00000000001;
 
    double learningRate = 0.2;
    double momentum = 0.0;
 
    // Inputs for xor problem
    //final double inputs[][] = { { 1, 1 }, { 1, 0 }, { 0, 1 }, { 0, 0 } };
    final double inputs[][] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    // Corresponding outputs, xor training data
    final double expectedOutputs[][] =  {{-1}, {1}, {1}, {-1}};
    //final double expectedOutputs[][] = { { 0 }, { 1 }, { 1 }, { 0 } };
    //double resultOutputs[][] = { { -1 }, { -1 }, { -1 }, { -1 } }; 
    double resultOutputs[][] = { { 0 }, { 0 }, { 0 }, { 0 } }; 
    
    private double[] input = new double[2 + 1];
    private double[] hidden = new double[4 + 1];
    private double[] output = new double[1];
    private int epoch;
    private double[] deltaOutput = new double[1];
    private double[] deltaHidden = new double[4];
    private double[][] weightHtoO = new double[5][1];
    private double[][] weightItoH = new double[3][4];
    private double[][] deltaweightItoH = new double[3][4];
    private double[][] deltaweightHtoO = new double[5][1];
    private double[] error = new double[1];
    private double[] err = new double[1];
    
   
 
    public NeuralNetwork(double learningRate, double momentum) {        
		this.learningRate = learningRate;
        this.momentum = momentum;                
        epoch = 0;
    }
 
    // random   
    public void setWeights() {
        
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 4; j++) {
            	weightItoH[i][j] = Math.random() - 0.5;
            }
        }        
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < 1; j++) {
            	weightHtoO[i][j] = Math.random() - 0.5;
            }
        }
    }

    /*public double sigmoid(double x) {
	//1.0 / (1.0 +  (Math.exp(-x)))
    return 1 / (1.0 +  (Math.exp(-x)));
}*/
    public double sigmoid(double x) {
		
	    return (double)(2) / (1 + Math.exp(-x)) -1;
	}  
    
    /**
     * Calculate the output of the neural network based on the input The forward
     * operation
     */
    public void activate(double[] sample) {
    	
    	for(int i = 0; i < sample.length; i++) {
            input[i] = sample[i];
        }
    	//Bias
        input[2] = 1;
        hidden[4] = 1;                
    	for(int j = 0; j < 4; j++) {          
            for(int i = 0; i < 3; i++) {                                    
            	hidden[j] += (weightItoH[i][j]*input[i]);            	
            }            
            hidden[j] = sigmoid(hidden[j]);           
    }
        for(int k = 0; k < 1; k++) {
            for(int j = 0; j < 5; j++) {                           
                output[k] += (weightHtoO[j][k]*hidden[j]);
            }
            output[k]= sigmoid(output[k]);
           
        }
    }    
   
 
    /**
     * all output propagate back
     * 
     * @param expectedOutput
     *            first calculate the partial derivative of the error with
     *            respect to each of the weight leading into the output neurons
     *            bias is also updated here
     */
    public void applyBackpropagation() {
 
        
        
        for (int k = 0; k < 1; k++) {
            double ak = output[k];
            
            deltaOutput[k] = 0;
            //deltaOutput[k] = ak * (1 - ak) * (err[k]);         
            deltaOutput[k] = 0.5 * (1 - Math.pow(ak, 2)) * err[k];
            
        }
        
        for(int k = 0; k < 1; k++) {
        
            for (int j = 0; j < 4 + 1; j++) {
            	double aj = hidden[j];   
            	deltaweightHtoO[j][k] = momentum * deltaweightHtoO[j][k]+learningRate * deltaOutput[k]* aj;
                
                weightHtoO[j][k] += deltaweightHtoO[j][k];
                
                
            }
            
        }
 
        // update weights for the hidden layer

        for(int j = 0; j < 4; j++) {        
        	double  aj = hidden[j];
            deltaHidden[j] = 0;
            for (int k = 0; k < 1; k++) {
                
                deltaHidden[j] += deltaOutput[k]*weightHtoO[j][k];
                        
            }
            //deltaHidden[j] = aj * (1 - aj) * deltaHidden[j];
            deltaHidden[j] = 0.5* (1 - Math.pow(aj, 2)) * deltaHidden[j];          
        }
       
        for(int j = 0; j < 4; j++) {
            for(int i = 0; i < 2 + 1; i++) {            
                double ai = input[i];             
                deltaweightItoH[i][j] = momentum * deltaweightItoH[i][j]+learningRate * deltaHidden[j]* ai;
                weightItoH[i][j] += deltaweightItoH[i][j];                                               
                
            }
            
        }
    }
    void run(int maxSteps, double minError)throws IOException {       
        epoch = 0;
        // Train neural network until minError reached or maxSteps exceeded
        double [] ErrorList = new double[50000];
        do {        
        	for(int k = 0; k < 1; k++) {
                error[k] = 0;
            }
            for (int p = 0; p < inputs.length; p++) {
                double[] sample = inputs[p];
 
                activate(sample);
 
                for(int k = 0; k < 1; k++) {
                	err[k] =  expectedOutputs[p][k] - output[k];
                    error[k] += Math.pow(err[k], 2);
                    resultOutputs[p] = output;  
                }
          
                applyBackpropagation();                
            }
            for(int k = 0; k < 1; k++) {
	            error[k] /= 2;
	            
            }
            ErrorList[epoch] = error[0];
            epoch ++;
        }while(error[0] > minError && epoch <10000);
        File file = new File("chart4.txt");
		FileWriter out = new FileWriter(file);
		for(int q = 0; q<epoch; q++) {
			out.write(ErrorList[q]+"\t");
			out.write("\r\n");
		}
		out.close();
        
        System.out.println("Sum of squared errors = " + error[0]);
        System.out.println("##### EPOCH " + epoch+"\n");
        if (epoch == maxSteps) {
            System.out.println("!Error training try again");
        } 
    }
     
    
    
}