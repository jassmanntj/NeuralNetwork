package device;

import Jama.Matrix;

import java.io.Serializable;

/**
 * DeviceConvolutionLayer
 *
 * @author Timothy Jassmann
 * @version 06/02/2015
 */
public class DeviceConvolutionLayer extends DeviceStructuredLayer implements Serializable {
    private Matrix weights[][];
    private Matrix bias;
    private int activation;
    private double a;
    private double dropout;

    /**
     * DeviceConvolutionLayer - A constructor for the device convolution layer
     *
     * @param weights The weight matrices
     * @param bias The bias matrix
     * @param activation The activation function
     * @param a The a value (for the PReLU activation)
     * @param dropout The percent dropout used
     */
    public DeviceConvolutionLayer(Matrix[][] weights, Matrix bias, int activation, double a, double dropout) {
        this.weights = weights;
        this.bias = bias;
        this.activation = activation;
        this.a = a;
        this.dropout = dropout;
    }

    /**
     * compute - Computes the output of the layer
     *
     * @param input The input matrices representing each channel of the input
     *
     * @return The output of the layer
     */
    public Matrix[] compute(Matrix[] input) {
        Matrix[] result = new Matrix[weights.length];
        for (int feature = 0; feature < weights.length; feature++) {
            Matrix res = new Matrix(input[0].getRowDimension() - weights[0][0].getRowDimension() + 1,
                                    input[0].getColumnDimension() - weights[0][0].getColumnDimension() + 1);
            for (int channel = 0; channel < weights[feature].length; channel++) {
                res.plusEquals(DeviceUtils.convolve(input[channel], weights[feature][channel], input[channel].getRowDimension(), input[channel].getColumnDimension()));
            }
            Matrix featureBias = new Matrix(res.getRowDimension(), res.getColumnDimension(), bias.get(feature, 0));
            result[feature] = DeviceUtils.activationFunction(activation, res.plus(featureBias), a).times(1 - dropout);
        }


        return result;
    }

}

