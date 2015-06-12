package device;

import Jama.Matrix;

import java.io.Serializable;

/**
 * DevicePoolingLayer
 *
 * @author Timothy Jassmann
 * @version 06/02/2015
 */
public class DevicePoolingLayer extends DeviceStructuredLayer implements Serializable {
    private int poolDimension;
    private int type;

    /**
     * DevicePoolingLayer - Constructor for the DevicePoolingLayer
     *
     * @param poolDimension The pooling dimension
     * @param type They type of pooling
     */
    public DevicePoolingLayer(int poolDimension, int type) {
        this.poolDimension = poolDimension;
        this.type = type;
    }

    /**
     * compute - computes the output of the pooling layer
     *
     * @param input Input matrices representing each channel of the input
     *
     * @return The output of the pooling layer
     */
    public Matrix[] compute(Matrix[] input) {
        Matrix[] result = new Matrix[input.length];
        switch(type) {
            case nn.PoolingLayer.MAX:
                for(int i = 0; i < input.length; i++) {
                    result[i] = maxPool(input[i]);
                }
                break;
            case nn.PoolingLayer.MEAN:
            default:
                for(int i = 0; i < input.length; i++) {
                    result[i] = meanPool(input[i]);
                }
        }
        return result;
    }

    /**
     * maxPool - performs the max pooling operation
     *
     * @param input Matrix to pool
     *
     * @return Result of max pooling on input
     */
    private Matrix maxPool(Matrix input) {
        int resultRows = input.getRowDimension() / poolDimension;
        int resultCols = input.getColumnDimension() / poolDimension;
        Matrix result = new Matrix(resultRows, resultCols);
        for(int poolRow = 0; poolRow < resultRows; poolRow++) {
            for(int poolColumn = 0; poolColumn < resultCols; poolColumn++) {
                double max = input.get(poolRow * poolDimension, poolColumn * poolDimension);
                for(int row = poolRow * poolDimension;
                        row < poolRow * poolDimension + poolDimension; row++) {
                    for(int col = poolColumn * poolDimension; col < poolColumn * poolDimension + poolDimension; col++) {
                        if(input.get(row, col) > max) max = input.get(row, col);
                    }
                }
                result.set(poolRow, poolColumn, max);
            }
        }
        return result;
    }

    /**
     * meanPool - performs the mean pooling operation
     *
     * @param input Matrix to pool
     *
     * @return Result of mean pooling on input
     */
    private Matrix meanPool(Matrix input) {
        int resultRows = input.getRowDimension() / poolDimension;
        int resultCols = input.getColumnDimension() / poolDimension;
        Matrix result = new Matrix(resultRows, resultCols);
        for(int poolRow = 0; poolRow < resultRows; poolRow++) {
            for(int poolColumn = 0; poolColumn < resultCols; poolColumn++) {
                double mean = 0;
                for(int row = poolRow * poolDimension; row < poolRow * poolDimension + poolDimension; row++) {
                    for(int col = poolColumn * poolDimension; col < poolColumn * poolDimension + poolDimension; col++) {
                        mean += input.get(row, col);
                    }
                }
                result.set(poolRow, poolColumn, mean / (poolDimension * poolDimension));
            }
        }
        return result;
    }
}
