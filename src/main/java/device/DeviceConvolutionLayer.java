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
    private Matrix theta[][];
    private Matrix bias;
    private int activation;
    private double a;
    private double dropout;

    /**
     * DeviceConvolutionLayer - A constructor for the device convolution layer
     *
     * @param theta The weight matrices
     * @param bias The bias matrix
     * @param activation The activation function
     * @param a The a value (for the PReLU activation)
     * @param dropout The percent dropout used
     */
    public DeviceConvolutionLayer(Matrix[][] theta, Matrix bias, int layer1, double a, double dropout) {
        this.theta = theta;
        this.bias = bias;
        this.activation = layer1;
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
        Matrix[] result = new Matrix[theta.length];
        for (int feature = 0; feature < theta.length; feature++) {
            Matrix res = new Matrix(input[0].getRowDimension() - theta[0][0].getRowDimension() + 1, input[0].getColumnDimension() - theta[0][0].getColumnDimension() + 1);
            for (int channel = 0; channel < theta[feature].length; channel++) {
                res.plusEquals(DeviceUtils.conv2d(input[channel], theta[feature][channel]));
            }

            result[feature] = DeviceUtils.activationFunction(activation, res.plus(new Matrix(res.getRowDimension(), res.getColumnDimension(), bias.get(feature, 0))), a).times(1 - dropout);
        }


        return result;
    }

}

