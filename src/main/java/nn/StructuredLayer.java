package nn;

import device.DeviceStructuredLayer;
import org.jblas.DoubleMatrix;

/**
 * StructuredLayer - A layer which computes on structured images (convolution or pooling)
 *
 * @author Timothy Jassmann
 * @version 06/16/2015
 */
public abstract class StructuredLayer {
    /**
     * compute - used to compute the output of the layer
     *
     * @param input input to the layer
     *
     * @return output of the layer
     */
    public abstract DoubleMatrix[][] compute(DoubleMatrix[][] input);

    /**
     * computeGradient - used to compute the gradient of the layer
     *
     * @param input input to the layer
     * @param output output of the layer
     * @param delta gradient propagated to the layer
     *
     * @return gradients of the layer
     */
    public abstract Gradients computeGradient(final DoubleMatrix[][] input, final DoubleMatrix[][] output,
                                              final DoubleMatrix delta[][]);

    /**
     * feedForward - used for feed forward pass of the network in training. Overwrite if different than compute.
     *
     * @param input input to the layer
     *
     * @return output of feed forward pass of network
     */
    public DoubleMatrix[][] feedForward(DoubleMatrix[][] input) {
        return compute(input);
    }

    /**
     * getDevice - returns the device equivalent of the layer
     *
     * @return device equivalent of layer
     */
    public abstract DeviceStructuredLayer getDevice();

    /**
     * gradientCheck - performs gradient checking of layer
     *
     * @param gradients computed Gradients of the layer
     * @param input input to the neural network
     * @param labels labels of the neural network
     * @param nn neural network
     * @param epsilon epsilon value for gradient checking
     *
     * @return double array containing the norm between numerical and analytical gradients for weights,
     *          bias, and a. These values should be very small
     */
    public abstract double[] gradientCheck(Gradients gradients, DoubleMatrix[][] input, DoubleMatrix labels,
                                           NeuralNetwork nn, double epsilon);

    /**
     * initializeParameters - initializes the parameters of the layer. Overwrite if needed.
     */
    public void initializeParameters(){}

    /**
     * updateWeights - updates the weights of the layer based on calculated Gradients
     *
     * @param gradients Gradients to use to update weights
     * @param momentum momentum to use for update
     * @param alpha learning rate to use for update
     *
     * @return gradient propagated through the layer
     */
    public abstract DoubleMatrix[][] updateWeights(Gradients gradients, double momentum, double alpha);
}