package linearML;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

public class TestClassifier {

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		
		long startTime = System.currentTimeMillis();
		
		double[][] trainImages = getImages(new File("train-images.idx3-ubyte"));
		String[] trainLabels = getLabels(new File("train-labels.idx1-ubyte"));
		
		System.out.println("Loaded training data " + ((System.currentTimeMillis()-startTime)/1000d));
		
		double[][] testImages = getImages(new File("t10k-images.idx3-ubyte"));
		String[] testLabels = getLabels(new File("t10k-labels.idx1-ubyte"));
		
		System.out.println("Loaded test data " + ((System.currentTimeMillis()-startTime)/1000d));
		
		//We now have a training dataset
		
		LinearRegressionClassifier classifier = new LinearRegressionClassifier(testImages[0].length, new String[] {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});
		classifier.train(trainImages, trainLabels);
		
		//Save trained classifier in file
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(new File("classifier.obj")));
		out.writeObject(classifier);
		out.close();
		
		System.out.println("Trained KNN Classifier " + ((System.currentTimeMillis()-startTime)/1000d));
		
		int good = 0, bad = 0;
		for(int i = 0; i < testImages.length; i++) {
			if(classifier.classify(testImages[i]).equals(testLabels[i]))
				good++;
			else
				bad++;
		}
		
		System.out.println("Done! " + ((System.currentTimeMillis()-startTime)/1000d));
		System.out.println(good + " good and " + bad + " bad.");
		System.out.println("Accuracy: " + ((double)good/(good + bad) * 100) + ", Error: " + ((double)bad/(good + bad) * 100));
		System.out.println("Completed in " + ((System.currentTimeMillis() - startTime)/1000d) + "seconds");
	}
	
	public static double[][] getImages(File file) throws IOException {
		double[][] images;
		DataInputStream input = new DataInputStream(new FileInputStream(file));
		
		input.readInt(); //Skip the "magic number"
		int n = input.readInt();
		int row = input.readInt();
		int col = input.readInt();
		
		images = new double[n][row*col];
		
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < row*col; j++) {
				images[i][j] = (255 - input.readUnsignedByte())/255d; //INVERTED the pixel in order to prevent singular matrices to occur
			}
		}
		
		input.close();
		return images;
	}
	
	public static String[] getLabels(File file) throws IOException {
		String[] labels;
		DataInputStream input = new DataInputStream(new FileInputStream(file));
		
		input.readInt(); //Skip the "magic number"
		int n = input.readInt();
		
		labels = new String[n];
		
		for(int i = 0; i < n; i++) {
			labels[i] = "" + input.readUnsignedByte();
		}
		
		input.close();
		return labels;
	}
}
