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
		
		//Preliminary sums:
		double[] pixelSums = new double[dimension];
		double[] labelSums = new double[classes.length];
		
		//Matrices used for calculations (equation of the form Ax = B, where A,B are matrices. the X is the matrix containing the thetas in this case)
		Matrix A = new Matrix(dimension, dimension); //Contains all the coefficients of the thetas after calculating the gradients
		Matrix B = new Matrix(dimension, 1); //Contains all constants (constants only occur when on label in the training set corresponds with the class for which the thetas are being calculated)
		
		//Compute all pixel sums first (the sum of all the pixels in the same position in all images)
		for(int i = 0; i < dimension; i++) {
			double temp = 0;
			for(int j = 0; j < data.length; j++) {
				temp += data[j][i];
			}
			pixelSums[i] = temp;
		}
		
		//Then the sum of the label values. (Note: if one of the labels never appears, the whole system collapses as one of the matrix equations will be homogeneous and the solution will be the trivial one)
		for(int i = 0; i < classes.length; i++) {
			double temp = 0;
			for(int j = 0; j < labels.length; j++) {
				if(classes[i].equals(labels[j])) {
					temp += 1;
				}
			}
			labelSums[i] = temp;
		}
		
		//Here's the fun part! Calculating the gradients and finding the thetas for all classes!
		for(int i = 0; i < classes.length; i++) {
			for(int j = 0; j < dimension; j++) { //j represents the current row in the matrix
				for(int k = 0; k < dimension; k++) { //k represents the current column in the matrix
					if(j == k) {
						A.set(j, k, 2 * pixelSums[j]);
					}
					else
						A.set(j, k, 2 * pixelSums[j] * pixelSums[k]);
				}
				
				B.set(j, 0, 2 * labelSums[i] * pixelSums[j]);
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
