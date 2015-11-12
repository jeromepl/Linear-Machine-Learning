package linearML;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import Jama.Matrix;

public class LinearRegressionClassifier implements Serializable {

	private static final long serialVersionUID = 7197882259463196104L;

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

	public void train(double[][] data, String[] labels) {
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
						if(j == k)
							tempA += data[l][j];
						else
							tempA += data[l][j] * data[l][k];
						
						if(k == 0) //Only need to calculate labels once per row
							tempB += data[l][j];
					}
					
					tempA  *= 2;
					A.set(j, k, tempA);
					System.out.println("Class: " + i + ". Position: " + j + ", " + k + ". Value: " + tempA);
					
					if(k == 0) { //Again, only calculate labels once per row (one constant per equation)
						tempB *= 2;
						B.set(j, 0, tempB);
					}
				}
			}
			
			//System.out.println(Arrays.deepToString(A.getArray()));
			
			Matrix x = A.solve(B);
			theta.setMatrix(0, dimension - 1, i, i, x);
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
