package linearML;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import Jama.Matrix;

public class LinearClassifier implements Serializable {

	private static final long serialVersionUID = 7197882259463196104L;

	private int dimension;
	private String[] classes;
	private Matrix theta; // The Matrix we want to build which contains the
							// unknowns

	public LinearClassifier(int dimension, String[] classes) {
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

		// LHS matrix, nxn. Filled up for every class matrix.
		double[][] tempMatrix = new double[dimension][dimension];
		// RHS matrix which, combined with the above matrix creates our linear
		// system of equations
		double[][] solveM = new double[dimension][1];

		// This array contains the eventual matrix of coefficients (functions)
		// which will be used to classify
		double[][] solution = new double[classes.length][dimension];

		// We create a matrix of coefficients for each of our classes.
		for (int cl = 0; cl < classes.length; cl++) {
			int sumT = 0; // used for  summing coefficients 
			int sumC =0; //used for summing constants

			// Compute the coefficients in each index, 
			// Add reasoning later

			for(int r =0;r<dimension;r++)
			{
				sumC =0;
				for(int c=0; c<dimension; c++)
				{
					sumT =0;
					for(int d=0;d<data.length;d++)
					{
						//Every column, within a row represents the coefficient of a single unknown. Since Differentiate this: sum((t_1x_1 +t_2x_2+...)^2) becomes
						// sum(2*x_i(t_1x_1 +t_2_x2)) where x_i is the coefficcient of the variable we differentiate (this is the row).  
						sumT =+ 2*data[d][c]*data[d][r];
						//RHS of the linear solution we are building. If the lavel of the current vector matches, then it has a 100% probability
						// of being this current class. Since we derived the function, every "1" will be multiplied by 2 (least square method) and the coefficient
						// of the variable being differentiated
						if (labels[d].equals(classes[cl]))
							sumC += 2*data[d][r];}
					}
					solution[c][r]=sumT;
				}
				solveM[r][0] = sumC;
			}
			
			// Super ugly way to create a matrix for both the RHS and LHS, solve
			// it, then extra the resulting nx1 matrix and put it in the
			// solution
			solution[cl] = (new Matrix(tempMatrix).solve(new Matrix(solveM)))
					.getArray()[0];

		}

		theta = new Matrix(solution);

	}

	public Map<String, Double> classDistribution(double[] data) {
		Map<String, Double> distribution = new HashMap<String, Double>();

		Matrix dataM = new Matrix(new double[][] { data });

		// Multiply the 1xn data matrix by the nxm theta matrix to result in our
		// result of 1x10 propabilities matrix
		Matrix distributionM = dataM.arrayTimes(theta);

		// Transform back to matrix
		double[] result = distributionM.getArray()[0];

		// fill the hasmap with the keys and propabilities
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
