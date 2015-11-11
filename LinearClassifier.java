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
	
	public String classify(int[] data) {
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
	
	public void train(int[][] data) {
		Matrix matrixToSolve = new Matrix(dimension, dimension);
	}
	
	public Map<String, Float> classDistribution(int[] data) {
		Map<String, Float> distribution = new HashMap<String, Float>();
		
		
		
		return distribution;
	}
	
	public int getDimension() {
		return dimension;
	}
	
	public String[] getClasses() {
		return classes;
	}

}
