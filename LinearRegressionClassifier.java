package linearML;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import Jama.Matrix;

public class LinearRegressionClassifier implements Serializable {

	private static final long serialVersionUID = 7197882259463196104L;
	
	private static int NUM_PASSES = 10;
	private static double LEARNING_RATE = 20;

	private int dimension;
	private String[] classes;
	private Matrix theta; // The Matrix we want to build which contains the unknowns

	public LinearRegressionClassifier(int dimension, String[] classes) {
		this.dimension = dimension;
		this.classes = classes;
		theta = new Matrix(dimension, classes.length); // Create an empty matrix
	}

	public String classify(double[] data) {
		Map<String, Double> distribution = classDistribution(data);

		double maxValue = -1;
		String max = "";

		for (String key : distribution.keySet()) {
			double temp = distribution.get(key);
			if (temp > maxValue) {
				maxValue = temp;
				max = key;
			}
		}

		return max;
	}

	public void train(double[][] data, String[] labels) throws FileNotFoundException, UnsupportedEncodingException {
		//Musts: data.length == labels.length. All data[i] == dimension. All labels must appear at least once.
		//labels means classes
		
		//Matrices used for calculations (equation of the form Ax = B, where A,B are matrices. the X is the matrix containing the thetas in this case)
		Matrix A = new Matrix(dimension, dimension); //Contains all the coefficients of the thetas after calculating the gradients
		Matrix B = new Matrix(dimension, 1); //Contains all constants (constants only occur when on label in the training set corresponds with the class for which the thetas are being calculated)
		
		//Here's the fun part! Calculating the gradients and finding the thetas for all classes!
		for(int i = 0; i < classes.length; i++) {
			for(int j = 0; j < dimension; j++) { //j represents the current row in the matrix
				for(int k = 0; k < dimension; k++) { //k represents the current column in the matrix
					
					double tempA = 0;
					double tempB = 0;
					
					for(int l = 0; l < data.length; l++) {
						
						tempA += data[l][j] * data[l][k];
						
						if(k == 0 && labels[l].equals(classes[i])) //Only need to calculate labels once per row
							tempB += data[l][j];
					}
					
					tempA  *= 2;
					A.set(j, k, tempA);
					
					if(k == 0) { //Again, only calculate labels once per row (one constant per equation)
						tempB *= 2;
						B.set(j, 0, tempB);
					}
				}
				System.out.println("Class: " + i + ". Position: " + j);
			}
			
			//System.out.println(Arrays.deepToString(A.getArray()));
			PrintWriter writer = new PrintWriter("class"+i+".txt", "UTF-8");
			writer.println(Arrays.deepToString(A.getArray()));
			writer.close();
			
			Matrix x = A.solve(B);
			theta.setMatrix(0, dimension - 1, i, i, x);
		}
		
	}
	
	public void trainUsingGradientDescent(double[][] data, String[] labels) { //Stochastic Gradient descent
		
		//Initialize the theta Matrix to some random values
		for(int i = 0; i < dimension; i++) {
			for(int j = 0; j < classes.length; j++) {
				theta.set(i, j, 0.5);
			}
		}
		
		double cost = 0;
		
		for(int i = 0; i < NUM_PASSES; i++) { //Do it n times to make sure it converged
			for(int j = 0; j < data.length; j++) { //For all training values
				for(int k = 0; k < dimension; k++) { //A gradient per theta
					
					Map<String, Double> distribution = classDistribution(data[j]);
					
					for(int l= 0; l < classes.length; l++) { //Different set of thetas for every class
						
						double current = theta.get(k, l);
						double gradient = 0, tempCost = 0;
						
						double h = distribution.get(classes[l]);
						gradient = data[j][k] * h;
						tempCost = h;
						
						if(labels[j].equals(classes[l])) {
							gradient -= data[j][k];
							tempCost -= 1;
						}
						
						gradient /= data.length;
						
						theta.set(k, l, current - LEARNING_RATE * gradient);
						//bias.set(0, l, currentBias - LEARNING_RATE * biasGradient);
						
						cost += Math.pow(tempCost, 2);
					}
				}
				
				if(j % 1000 == 0) {
					System.out.println("Pass: " + (i + 1) + " of " + NUM_PASSES + ", Data no. " + (j+1) + ", Cost: " + cost / 1000);
					cost = 0;
				}
			}
		}
	}

	public Map<String, Double> classDistribution(double[] data) {
		Map<String, Double> distribution = new HashMap<String, Double>();

		Matrix dataM = new Matrix(new double[][] { data });

		// Multiply the 1xn data matrix by the nxm theta matrix to result in our
		// result of 1x10 probabilities matrix
		Matrix distributionM = dataM.times(theta);

		// Transform back to matrix
		double[] result = distributionM.getArray()[0];

		// fill the hashmap with the keys and probabilities
		for (int i = 0; i < classes.length; i++) {
			distribution.put(classes[i], result[i]);
		}

		return distribution;
	}

	public int getDimension() {
		return dimension;
	}

	public String[] getClasses() {
		return classes;
	}

	public Matrix getTheta() {
		return theta;
	}

}
