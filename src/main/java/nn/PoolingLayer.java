package nn;

import device.DeviceStructuredLayer;
import device.DevicePoolingLayer;
import org.jblas.DoubleMatrix;

/**
 * PoolingLayer - Structured Layer that performs pooling operation
 *
 * @author Timothy Jassmann
 * @version 06/16/2015
 */
public class PoolingLayer extends StructuredLayer {
    private int poolDim;
    private int type;
    private int stride;
    public static final int MAX = 0;
    public static final int MEAN = 1;

    /**
     * PoolingLayer - constructor for PoolingLayer
     *
     * @param poolDim dimension to pool by
     * @param type type of pooling
     */
    public PoolingLayer(int poolDim, int type, int stride) {
        this.poolDim = poolDim;
        this.type = type;
        this.stride = stride;
    }

    /**
     * compute - used to compute the output of the layer
     *
     * @param input input to the layer
     *
     * @return output of the layer
     */
    public DoubleMatrix[][] compute(DoubleMatrix[][] input) {
        DoubleMatrix[][] result = new DoubleMatrix[input.length][input[0].length];
        switch(type) {
            case MAX:
                for(int i = 0; i < input.length; i++) {
                    for(int j = 0; j < input[i].length; j++) {
                        result[i][j] = maxPool(input[i][j]);
                    }
                }
                break;
            case MEAN:
            default:
                for(int i = 0; i < input.length; i++) {
                    for(int j = 0; j < input[i].length; j++) {
                        result[i][j] = meanPool(input[i][j]);
                    }
                }
        }
        return result;
    }

    /**
     * computeGradient - used to compute the gradient of the layer
     *
     * @param input input to the layer
     * @param output output of the layer
     * @param delta gradient propagated to the layer
     *
     * @return gradients of the layer
     */
    public Gradients computeGradient(DoubleMatrix[][] input, DoubleMatrix[][] output, DoubleMatrix delta[][]) {
        DoubleMatrix[][] result = new DoubleMatrix[delta.length][delta[0].length];
        switch(type) {
            case MAX:
                for(int i = 0; i < delta.length; i++) {
                    for(int j = 0; j < delta[i].length; j++) {
                        result[i][j] = maxExpand(delta[i][j], input[i][j]);
                    }
                }
                break;
            case MEAN:
            default:
                for(int i = 0; i < delta.length; i++) {
                    for(int j = 0; j < delta[i].length; j++) {
                        result[i][j] = meanExpand(delta[i][j]);
                    }
                }
        }
        return new Gradients(null, null, result, 0);
    }

    /**
     * getDevice - returns the device equivalent of the layer
     *
     * @return device equivalent of layer
     */
    public DeviceStructuredLayer getDevice() {
        return new DevicePoolingLayer(poolDim, type, stride);
    }

    /**
     * gradientCheck - performs gradient checking of layer
     *
     * @param gradients computed Gradients of the layer
     * @param input input to the neural network
     * @param labels labels of the neural network
     * @param nn neural network
     * @param epsilon epsilon value for gradient checking
     *
     * @return there are no gradients to check for this layer. Returns empty array
     */
    public double[] gradientCheck(Gradients gradients, DoubleMatrix[][] input, DoubleMatrix labels,
                                  NeuralNetwork nn, double epsilon) {
        return new double[0];
    }

    /**
     * updateWeights - No weights to update - just returns gradient propagated through layer
     *
     * @param gradients Gradients to use to update weights
     * @param momentum momentum to use for update
     * @param alpha learning rate to use for update
     *
     * @return gradient propagated through the layer
     */
    public DoubleMatrix[][] updateWeights(Gradients gradients, double momentum, double alpha) {
        return gradients.getDelt();
    }

    /**
     * maxExpand - expands the gradient propageted into the layer if max pooling layer
     *
     * @param gradient gradient propagated into layer
     * @param input input to layer
     *
     * @return expanded gradient
     */
    private DoubleMatrix maxExpand(DoubleMatrix gradient, DoubleMatrix input) {
        DoubleMatrix expandedMatrix = new DoubleMatrix(gradient.rows * stride, gradient.columns * stride);
        int offset = (poolDim - stride) / 2;
        for (int i = 0; i < gradient.rows; i++) {
            for (int j = 0; j < gradient.columns; j++) {
                int startrow = (i * stride - offset);
                int startcol = (j * stride - offset);
                int endrow = (startrow + poolDim) < input.rows ? (startrow + poolDim):(input.rows);
                int endcol = (startcol + poolDim) < input.columns ? (startcol + poolDim):(input.columns);
                startrow = startrow > 0 ? startrow:0;
                startcol = startcol > 0 ? startcol:0;
                DoubleMatrix patch = input.getRange(startrow, endrow, startcol, endcol);
                int index = patch.argmax();
                int row = startrow + (index % patch.rows);
                int col = startcol + (index / patch.rows);
                expandedMatrix.put(row, col, expandedMatrix.get(row, col) + gradient.get(i, j));
            }
        }
        return expandedMatrix;
    }

    /**
     * maxPool - performs max pooling on the input
     *
     * @param input matrix to perform max pooling on
     *
     * @return pooled matrix
     */
    private DoubleMatrix maxPool(DoubleMatrix input) {
        int resultRows = input.rows / stride;
        int resultCols = input.columns / stride;
        int offset = (poolDim - stride) / 2;
        DoubleMatrix result = new DoubleMatrix(resultRows, resultCols);
        for(int poolRow = 0; poolRow < resultRows; poolRow++) {
            for(int poolCol = 0; poolCol < resultCols; poolCol++) {
                int startrow = (poolRow * stride - offset);
                int startcol = (poolCol * stride - offset);
                int endrow = startrow + poolDim;
                int endcol = startcol + poolDim;
                DoubleMatrix patch = input.getRange(startrow > 0 ? startrow:0, endrow < input.rows ? endrow:(input.rows),
                        startcol > 0 ? startcol:0, endcol < input.columns ? endcol:(input.columns));
                result.put(poolRow, poolCol, patch.max());
            }
        }
        return result;
    }

    /**
     * meanExpand - expands the gradient propageted into the layer if max pooling layer
     *
     * @param gradient gradient propagated into layer
     *
     * @return expanded gradient
     */
    private DoubleMatrix meanExpand(DoubleMatrix gradient) {
        DoubleMatrix expandedMatrix = new DoubleMatrix(gradient.rows * stride, gradient.columns * stride);
        int offset = (poolDim - stride) / 2;
        for (int i = 0; i < gradient.rows; i++) {
            for (int j = 0; j < gradient.columns; j++) {
                int startrow = (i * stride - offset);
                int startcol = (j * stride - offset);
                int endrow = (startrow + poolDim) < expandedMatrix.rows ? (startrow + poolDim):(expandedMatrix.rows);
                int endcol = (startcol + poolDim) < expandedMatrix.columns ? (startcol + poolDim):(expandedMatrix.columns);
                startrow = startrow > 0 ? startrow:0;
                startcol = startcol > 0 ? startcol:0;
                double scale = (endrow - startrow) * (endcol - startcol);
                double value = gradient.get(i, j) / scale;
                for (int k = startrow; k < endrow; k++) {
                    for (int l = startcol; l < endcol; l++) {
                        expandedMatrix.put(k, l, value + expandedMatrix.get(k, l));
                    }
                }
            }
        }
        return expandedMatrix;
    }

    /**
     * maxPool - performs mean pooling on the input
     *
     * @param input matrix to perform mean pooling on
     *
     * @return pooled matrix
     */
    private DoubleMatrix meanPool(DoubleMatrix input) {
        int resultRows = input.rows / stride;
        int resultCols = input.columns / stride;
        int offset = (poolDim - stride) / 2;
        DoubleMatrix result = new DoubleMatrix(resultRows, resultCols);
        for(int poolRow = 0; poolRow < resultRows; poolRow++) {
            for(int poolCol = 0; poolCol < resultCols; poolCol++) {
                int startrow = (poolRow * stride - offset);
                int startcol = (poolCol * stride - offset);
                int endrow = (startrow + poolDim) < input.rows ? (startrow + poolDim):(input.rows);
                int endcol = (startcol + poolDim) < input.columns ? (startcol + poolDim):(input.columns);
                startrow = startrow > 0 ? startrow:0;
                startcol = startcol > 0 ? startcol:0;
                DoubleMatrix patch = input.getRange(startrow, endrow, startcol, endcol);
                result.put(poolRow, poolCol, patch.mean());
            }
        }
        return result;
    }
}
