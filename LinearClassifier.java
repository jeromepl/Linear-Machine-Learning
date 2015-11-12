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
	
	public void train(double[][] data, String[] labels) {

	//LHS matrix, nxn. Filled up for every class matrix. 
	double[][] tempMatrix = new double[dimension][dimension];
	//RHS matrix which, combined with the above matrix creates our linear system of equations
	double[][] solveM = new double [dimension][1];

	//fill this up
	double[][] solution = new double[classes.length][dimensions];

	//We create a matrix of coefficients for each of our classes. 
	for(int cl=0; cl<classes.length;cl++){
		int sum =0; //used for both summing coefficients and summing constants
		
		//Compute the coefficients in each index, going by column.
		//Add reasoning later
		//column
		for(int a=0; a<dimension;a++){
			//rows
			for(int i = 0; i<dimension;i++)
			{
				sum =0;
				//bytes
				for(int j=0; j<dimension;j++)
				{
					//Diagonal of the matrix, square of the entry
					if(i==a)
						sum+= 2*(data[j][a]*data[j][a]);
					else
						sum+= data[j][a];
				}
				tempMatrix[a][i] =sum;
			}
		}
		 

			//Compute constants
			for(int r=0; r<dimension;r++)
			{
				sum =0;	
				for(double x: data[r]){
				if(labels[r].equals(classes[cl]))
					sum+= x;
				}
			//Fill up the RHS matrix 
			solveM[r][0] =sum;	
			}


		//Super ugly way to create a matrix for both the RHS and LHS, solve it, then extra the resulting nx1 matrix and put it in the solution
		solution[cl] = (new Matrix(tempMatrix).solve(new Matrix(solveM))).getArray()[0];	
		
		}

	

	theta = new Matrix(solution);

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
