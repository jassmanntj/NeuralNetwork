package nn;

import org.jblas.DoubleMatrix;

import java.util.HashMap;

/**
 * Loader - Loads data into matrices to be used by network
 *
 * @author Timothy Jassmann
 * @version 06/16/2015
 */
public abstract class Loader {
    /**
     * getLabelMap returns mapping of string labels to numerical labels
     *
     * @return mapping of string labels to numerical labels
     */
    public abstract HashMap<String, Double> getLabelMap();

    /**
     * getTestData - returns test set given batch size and number
     *
     * @param batch batch number for test set
     * @param numBatches number of batches to split into
     *
     * @return test data
     */
    public abstract DoubleMatrix[][] getTestData(int batch, int numBatches);

    /**
     * getTestLabels - returns test labels given batch size and number
     *
     * @param batch batch number for test set
     * @param numBatches number of batches to split into
     *
     * @return test labels
     */
    public abstract DoubleMatrix getTestLabels(int batch, int numBatches);

    /**
     * getTrainData - returns train set given batch size and number
     *
     * @param batch batch number of test set
     * @param numBatches number of batches to split into
     *
     * @return train data
     */
    public abstract DoubleMatrix[][] getTrainData(int batch, int numBatches);

    /**
     * getTrainLabels - returns train labels given batch size and number
     *
     * @param batch batch number of test set
     * @param numBatches number of batches to split into
     *
     * @return train labels
     */
    public abstract DoubleMatrix getTrainLabels(int batch, int numBatches);
}
