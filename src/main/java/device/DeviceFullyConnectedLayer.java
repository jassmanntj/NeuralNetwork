package device;

import Jama.Matrix;

import java.io.Serializable;

/**
 * DeviceFullyConnectedLayer
 *
 * @author Timothy Jassmann
 * @version 06/02/2015
 */
public class DeviceFullyConnectedLayer implements Serializable {
    private int activation;
    private double a;
    private Matrix weights;
    private Matrix bias;
    private double dropout;

    /**
     * DeviceFullyConnectedLayer - a constructor for the device fully connected layer
     *
     * @param weights Weight matrix for the layer
     * @param bias Bias matrix for the layer
     * @param activation  Activation function for the layer
     * @param a The a value (for the PReLU activation)
     * @param dropout The percent dropout used
     */
    public DeviceFullyConnectedLayer(Matrix weights, Matrix bias, int activation, double a, double dropout) {
        this.activation = activation;
        this.a = a;
        this.weights = weights;
        this.bias = bias;
        this.dropout = dropout;
    }

    /**
     * compute - Computes the output of the layer
     *
     * @param input The input matrix (row vector)
     *
     * @return The output of the layer (row vector)
     */
    public Matrix compute(Matrix input) {
        Matrix result = input.times(weights);
        result.plusEquals(bias);
        return DeviceUtils.activationFunction(activation, result, a).times(1 - dropout);
    }
}
