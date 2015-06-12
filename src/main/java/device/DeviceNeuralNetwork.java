package device;

import Jama.Matrix;

import java.io.Serializable;

/**
 * DeviceNeuralNetwork
 *
 * @author Timothy Jassmann
 * @version 06/02/2015
 */
public class DeviceNeuralNetwork implements Serializable{
    DeviceStructuredLayer[] structuredLayers;
    DeviceFullyConnectedLayer[] fullyConnectedLayers;

    /**
     * DeviceNeuralNetwork - Constructor for the device neural network
     *
     * @param structuredLayers The convolution and pooling layers of the network
     * @param fullyConnectedLayers The fully connected layers of the network
     */
    public DeviceNeuralNetwork(DeviceStructuredLayer[] structuredLayers,
                               DeviceFullyConnectedLayer[] fullyConnectedLayers) {
        this.structuredLayers = structuredLayers;
        this.fullyConnectedLayers = fullyConnectedLayers;
    }

    /**
     * compute - computes the output of the network
     *
     * @param input Input matrices representing each channel of the input
     *
     * @return The output of the network
     */
    public Matrix compute(Matrix[] input) {
        for (DeviceStructuredLayer structuredLayer : structuredLayers) {
            input = structuredLayer.compute(input);
        }
        Matrix in = DeviceUtils.flatten(input);
        for (DeviceFullyConnectedLayer fullyConnectedLayer : fullyConnectedLayers) {
            in = fullyConnectedLayer.compute(in);
        }
        return in;
    }
}
