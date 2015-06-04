package nn;

import device.DeviceStructuredLayer;
import device.DevicePoolingLayer;
import org.jblas.DoubleMatrix;

import java.io.BufferedWriter;
import java.io.IOException;

/**
 * Created by Tim on 4/1/2015.
 */
public class PoolingLayer extends StructuredLayer {
    private int poolDim;
    private int type;
    public static final int MAX = 0;
    public static final int MEAN = 1;
    public PoolingLayer(int poolDim, int type) {
        this.poolDim = poolDim;
        this.type = type;
    }

    public DoubleMatrix[][] compute(DoubleMatrix[][] in) {
        DoubleMatrix[][] result = new DoubleMatrix[in.length][in[0].length];
        switch(type) {
            case MAX:
                for(int i = 0; i < in.length; i++) {
                    for(int j = 0; j < in[i].length; j++) {
                        result[i][j] = maxPool(in[i][j]);
                    }
                }
                break;
            case MEAN:
            default:
                for(int i = 0; i < in.length; i++) {
                    for(int j = 0; j < in[i].length; j++) {
                        result[i][j] = pool(in[i][j]);
                    }
                }
        }
        return result;
    }

    public DoubleMatrix[][] backpropagation(Gradients cr, double momentum, double alpha) {
        return cr.delt;
    }

    public Gradients cost(final DoubleMatrix[][] input, final DoubleMatrix[][] output, final DoubleMatrix delta[][]) {
        DoubleMatrix[][] result = new DoubleMatrix[delta.length][delta[0].length];
        switch(type) {
            case MAX:
                for(int i = 0; i < delta.length; i++) {
                    for(int j = 0; j < delta[i].length; j++) {
                        result[i][j] = expandMax(delta[i][j], input[i][j]);
                    }
                }
                break;
            case MEAN:
            default:
                for(int i = 0; i < delta.length; i++) {
                    for(int j = 0; j < delta[i].length; j++) {
                        result[i][j] = expand(delta[i][j]);
                    }
                }
        }
        return new Gradients(null, null, result, 0);
    }

    public DoubleMatrix[][] gradientCheck(Gradients cr, DoubleMatrix[][] in, DoubleMatrix labels, NeuralNetwork cnn) {
        return cr.delt;
    }

    private DoubleMatrix pool(DoubleMatrix convolvedFeature) {
        int resultRows = convolvedFeature.rows/poolDim;
        int resultCols = convolvedFeature.columns/poolDim;
        DoubleMatrix result = new DoubleMatrix(resultRows, resultCols);
        for(int poolRow = 0; poolRow < resultRows; poolRow++) {
            for(int poolCol = 0; poolCol < resultCols; poolCol++) {
                DoubleMatrix patch = convolvedFeature.getRange(poolRow*poolDim, poolRow*poolDim+poolDim, poolCol*poolDim, poolCol*poolDim+poolDim);
                result.put(poolRow, poolCol, patch.mean());
            }
        }
        return result;
    }

    private DoubleMatrix maxPool(DoubleMatrix convolvedFeature) {
        int resultRows = convolvedFeature.rows/poolDim;
        int resultCols = convolvedFeature.columns/poolDim;
        DoubleMatrix result = new DoubleMatrix(resultRows, resultCols);
        for(int poolRow = 0; poolRow < resultRows; poolRow++) {
            for(int poolCol = 0; poolCol < resultCols; poolCol++) {
                DoubleMatrix patch = convolvedFeature.getRange(poolRow*poolDim, poolRow*poolDim+poolDim, poolCol*poolDim, poolCol*poolDim+poolDim);
                result.put(poolRow, poolCol, patch.max());
                //poolArgs += patch.argmax()+" ";
            }
        }
        return result;
    }

    private DoubleMatrix expandMax(DoubleMatrix in, DoubleMatrix orig) {
        if(poolDim > 1) {
            DoubleMatrix expandedMatrix = new DoubleMatrix(in.rows * poolDim, in.columns * poolDim);
            double scale = (poolDim * poolDim);
            for (int i = 0; i < in.rows; i++) {
                for (int j = 0; j < in.columns; j++) {
                    DoubleMatrix patch = orig.getRange(i*poolDim, i*poolDim+poolDim, j*poolDim, j*poolDim+poolDim);
                    int index = patch.argmax();
                    int row = i*poolDim + index%poolDim;
                    int col = j * poolDim + index/poolDim;
                    expandedMatrix.put(row, col, in.get(i, j));
                    //expandArgs += patch.argmax()+" ";
                }
            }
            return expandedMatrix;
        }
        else return in;
    }

    private DoubleMatrix expand(DoubleMatrix in) {
        if(poolDim > 1) {
            DoubleMatrix expandedMatrix = new DoubleMatrix(in.rows * poolDim, in.columns * poolDim);
            double scale = (poolDim * poolDim);
            for (int i = 0; i < in.rows; i++) {
                for (int j = 0; j < in.columns; j++) {
                    double value = in.get(i, j) / scale;
                    for (int k = 0; k < poolDim; k++) {
                        for (int l = 0; l < poolDim; l++) {
                            expandedMatrix.put(i * poolDim + k, j * poolDim + l, value);
                        }
                    }
                }
            }
            return expandedMatrix;
        }
        else return in;
    }

    public void writeLayer(BufferedWriter writer) {
        try {
            writer.write(Utils.POOLLAYER+"\n");
            writer.write(poolDim+"\n");
        }
        catch(IOException e) {
            e.printStackTrace();
        }
    }

    public DeviceStructuredLayer getDevice() {
        return new DevicePoolingLayer(poolDim, type);
    }
}
