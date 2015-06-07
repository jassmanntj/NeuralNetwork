package nn;

import org.jblas.DoubleMatrix;

import java.util.HashMap;

/**
 * Created by jassmanntj on 6/4/2015.
 */
public abstract class Loader {
    public abstract DoubleMatrix[][] getTrainData(int batch, int batchSize);
    public abstract DoubleMatrix getTrainLabels(int batch, int batchSize);
    public abstract DoubleMatrix[][] getTestData(int batch, int batchSize);
    public abstract DoubleMatrix getTestLabels(int batch, int batchSize);
    public abstract HashMap<String, Double> getLabelMap();
}
