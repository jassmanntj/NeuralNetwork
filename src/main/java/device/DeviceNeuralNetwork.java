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
    DeviceStructuredLayer[] cls;
    DeviceFullyConnectedLayer[] fcs;

    /**
     * DeviceNeuralNetwork - Constructor for the device neural network
     *
     * @param cls The convolution and pooling layers of the network
     * @param fcs The fully connected layers of the network
     */
    public DeviceNeuralNetwork(DeviceStructuredLayer[] cls, DeviceFullyConnectedLayer[] fcs) {
        this.cls = cls;
        this.fcs = fcs;
    }

    /**
     * compute - computes the output of the network
     *
     * @param input Input matrices representing each channel of the input
     *
     * @return The output of the network
     */
    public Matrix compute(Matrix[] input) {
        for(int i = 0; i < cls.length; i++) {
            input = cls[i].compute(input);
        }
        Matrix in = DeviceUtils.flatten(input);
        for(int i = 0; i < fcs.length; i++) {
            in = fcs[i].compute(in);
        }
        return in;
    }
}
