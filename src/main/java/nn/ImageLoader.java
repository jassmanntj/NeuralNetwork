package nn;

import org.imgscalr.Scalr;
import org.imgscalr.Scalr.Method;
import org.imgscalr.Scalr.Rotation;
import org.jblas.DoubleMatrix;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * ImageLoader - implementation of loader class for images
 *
 * @author Timothy Jassmann
 * @version 06/16/2015
 */
@SuppressWarnings("UnusedDeclaration")
public class ImageLoader extends Loader {
    DoubleMatrix imgArr[][];
    DoubleMatrix lbls;
	int channels;
	int width, height;
    int[] counts;
	HashMap<String, Double> labelMap;
	ArrayList<String> names;

    /**
     * ImageLoader - Constructor for ImageLoader class. Loads folder structure into DoubleMatrix.
     *          Folder should contain one folder for each classification. Each folder contains
     *          images of that classification of leaves. All images should be the same aspect ratio.
     *
     * @param folder folder to load data from
     * @param channels number of channels of data in images (should probably always be 3)
     * @param width width to rescale images to
     * @param height height to rescale images to
     */
    public ImageLoader(File folder, int channels, int width, int height) {
        int position = 0;
        this.channels = channels;
        this.width = width;
        this.height = height;
        this.labelMap = constructLabelMap(folder);
        this.names = new ArrayList<String>();
        this.counts = countImages(folder, labelMap);
        int total = 0;
        for(int c : counts) {
            total += c;
        }
        this.lbls = new DoubleMatrix(total,labelMap.size());
        this.imgArr = new DoubleMatrix[total][3];
        int classification = 0;
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(File leaf : folder.listFiles()) {
            if(leaf.isDirectory() && labelMap.containsKey(leaf.getName())) {
                for(File image : leaf.listFiles()) {
                    Runnable ip = new ImageProcessor(image, classification, position);
                    executor.execute(ip);
                    position += counts[classification];
                    position = position%total + position/total;
                }
            }
            classification++;
        }
        executor.shutdown();
        while(!executor.isTerminated());
    }

    /**
     * ImageProcessor - class for processing images. Rotates and scales images to the correct size.
     */
    private class ImageProcessor implements Runnable {
        private File image;
        private double classification;
        private int position;

        /**
         * ImageProcessor - constructor for the ImageProcessor subclass
         *
         * @param image image to process
         * @param classification classification of leaf
         * @param position position to store leaf data in array
         */
        public ImageProcessor(File image, double classification, int position) {
            this.image = image;
            this.position = position;
            this.classification = classification;
        }

        /**
         * run - executes the ImageProcessor - loading image, rotating it, and scaling it
         */
        public void run(){
            System.out.println(position);
            BufferedImage img = null;
            try {
                img = ImageIO.read(image);
            } catch (IOException e) {
                e.printStackTrace();
            }
            if ( (img.getHeight()-img.getWidth())*(height-width) < 0) {
                img = Scalr.rotate(img, Rotation.CW_90);
            }
            else {
                img = Scalr.rotate(img, Rotation.CW_180);
                img = Scalr.rotate(img, Rotation.CW_180);
            }
            img = Scalr.resize(img, Method.QUALITY, width, height);
            int[] pixels = ((DataBufferInt) img.getRaster().getDataBuffer()).getData();
            img.flush();
            for(int i = 0; i < channels; i++) {
                imgArr[position][i] = new DoubleMatrix(height, width);
            }
            if (pixels.length == width * height) {
                for (int i = 0; i < pixels.length; i++) {
                    for (int j = 0; j < channels; j++) {
                        imgArr[position][j].put(i/width, i%width, ((pixels[i] >>> (8 * j)) & 0xFF));
                    }
                }
                lbls.put(position, (int)classification, 1);

            }
        }
    }

    /**
     * getData - returns the normalized entire data set
     *
     * @return normalized entire data set
     */
    public DoubleMatrix[][] getData() {
        return Utils.normalizeData(imgArr);
    }

    /**
     * getLabels - returns the labels of the entire data set
     *
     * @return labels of entire data set
     */
    public DoubleMatrix getLabels() {
        return lbls;
    }

    /**
     * getLabelMap returns mapping of string labels to numerical labels
     *
     * @return mapping of string labels to numerical labels
     */
    public HashMap<String, Double> getLabelMap() {
        return labelMap;
    }

    /**
     * getTestData - returns test set given batch size and number
     *
     * @param batch batch number for test set
     * @param numBatches batch size
     *
     * @return test data
     */
    public DoubleMatrix[][] getTestData(int batch, int numBatches) {
        int batchSize = imgArr.length/numBatches;
        DoubleMatrix[][] images = new DoubleMatrix[batchSize][];
        int k = 0;
        for(int j = 0; j < imgArr.length; j++) {
            if(j >= batch*batchSize && j < batch*batchSize+batchSize) {
                images[k++] = imgArr[j];
            }
        }
        return Utils.normalizeData(images);
    }

    /**
     * getTestLabels - returns test labels given batch size and number
     *
     * @param batch batch number for test set
     * @param numBatches number of batches to split into
     *
     * @return test labels
     */
    public DoubleMatrix getTestLabels(int batch, int numBatches) {
        int batchSize = imgArr.length/numBatches;
        DoubleMatrix labs = null;
        for(int j = 0; j < lbls.rows; j++) {
            if(j >= batch*batchSize && j < batch*batchSize+batchSize) {
                if (labs == null) {
                    labs = lbls.getRow(j);
                }
                else {
                    labs = DoubleMatrix.concatVertically(labs, lbls.getRow(j));
                }
            }
        }
        return labs;
    }

    /**
     * getTrainData - returns train set given batch size and number
     *
     * @param batch batch number of test set
     * @param batchSize batch size
     *
     * @return train data
     */
    public DoubleMatrix[][] getTrainData(int batch,  int numBatches) {
        int batchSize = imgArr.length/numBatches;
        DoubleMatrix[][] images = new DoubleMatrix[(imgArr.length-batchSize)][];
        int k = 0;
        for(int j = 0; j < imgArr.length; j++) {
            if(j < batch*batchSize || j >= batch*batchSize+batchSize) {
                images[k++] = imgArr[j];
            }
        }
        return Utils.normalizeData(images);
    }

    /**
     * getTrainLabels - returns train labels given batch size and number
     *
     * @param batch batch number of test set
     * @param numBatches number of batches to split into
     *
     * @return train labels
     */
    public DoubleMatrix getTrainLabels(int batch, int numBatches) {
        int batchSize = imgArr.length/numBatches;
        DoubleMatrix labs = null;
        for(int j = 0; j < lbls.rows; j++) {
            if(j < batch*batchSize || j >= batch*batchSize+batchSize) {
                if (labs == null) {
                    labs = lbls.getRow(j);
                }
                else {
                    labs = DoubleMatrix.concatVertically(labs, lbls.getRow(j));
                }
            }
        }
        return labs;
    }

    /**
     * constructLabelMap - constructs a mapping of string labels to numerical labels
     *
     * @param folder folder to construct label map from
     *
     * @return labelmap
     */
	private HashMap<String, Double> constructLabelMap(File folder) {
		HashMap<String, Double> labelMap = new HashMap<String, Double>();
		double labelNo = -1;
		for(File leaf : folder.listFiles()) {
			labelNo++;
			leaf.listFiles();
			labelMap.put(leaf.getName(), labelNo);
		}
		return labelMap;
	}

    /**
     * countImages - counts images in each subfolder
     *
     * @param folder folder to count images in
     * @param labelMap mapping of string labels to numerical labels
     *
     * @return counts of each image
     */
    private int[] countImages(File folder, HashMap<String, Double> labelMap) {
        int count[] = new int[labelMap.size()];
        for(File leaf : folder.listFiles()) {
            if (leaf.isDirectory() && labelMap.containsKey(leaf.getName())) {
                for (File image : leaf.listFiles()) {
                    count[labelMap.get(leaf.getName()).intValue()]++;
                }
            }
        }
        return count;
    }
}
