package device;

import Jama.Matrix;

/**
 * Device2DLayer - a wrapper for layers with 2D inputs and outputs
 *          (convolution and pooling layers)
 *
 * @author Timothy Jassmann
 * @version 06/02/2015
 */
public abstract class DeviceStructuredLayer {
    /**
     * compute - computes the output of the layer
     *
     * @param input Input matrices representing each channel of the input
     *
     * @return The output of the layer
     */
    public abstract Matrix[] compute(Matrix[] input);
}
