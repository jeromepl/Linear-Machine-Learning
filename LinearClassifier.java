package linearML;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import Jama.Matrix;

public class LinearClassifier implements Serializable {

	private static final long serialVersionUID = 7197882259463196104L;
	
	private int dimension;
	private String[] classes;
	private Matrix theta; //The Matrix we want to build which contains the unknowns
	
	public LinearClassifier(int dimension, String[] classes) {
		this.dimension = dimension;
		this.classes = classes;
		theta = new Matrix(dimension, classes.length); //Create an empty matrix
	}
	
	public String classify(double[] data) {
		Map<String, Float> distribution = classDistribution(data);
		
		float maxValue = -1;
		String max = "";
		
		for(String key: distribution.keySet()) {
			float temp = distribution.get(key);
			if(temp > maxValue) {
				maxValue = temp;
				max = key;
			}
		}
		
		return max;
	}
	
	public void train(double[][] data) {
		Matrix matrixToSolve = new Matrix(dimension, dimension);
	}
	
	public Map<String, Float> classDistribution(double[] data) {
		Map<String, Float> distribution = new HashMap<String, Float>();

		Matrix dataM = new Matrix(new double[][]{data});
		//Multiply the 1xn data matrix by the nxm theta matrix to result in our result of 1x10 propabilities matrix
		Matrix distributionM = dataM.arrayTimes(theta);
		//Transform back to matrix
		double[] result = distributionM.getArray()[0];
		//fill the hasmap with the keys and propabilities
		for(int i=0;i<classes.length; i++)
		{
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

}
