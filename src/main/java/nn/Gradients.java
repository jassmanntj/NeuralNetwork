package nn;

import org.jblas.DoubleMatrix;

/**
 * Gradients - Encapsulates gradients of layer for backpropagation algorithm
 *
 * @author Timothy Jassmann
 * @version 06/16/2015
 */
public class Gradients {
    private DoubleMatrix delta;
    private DoubleMatrix weightGrad;
    private DoubleMatrix biasGrad;
    private DoubleMatrix[][] delt;
    private DoubleMatrix[][] wGrad;
    private double aGrad;

    /**
     * Gradients - Constructor for gradients class (used in FullyConnectedLayers)
     *
     * @param weightGrad gradient of the weight matrix
     * @param biasGrad gradient of the bias vector
     * @param delta gradient propagated through the layer
     * @param aGrad gradient of the a value
     */
    public Gradients(DoubleMatrix weightGrad, DoubleMatrix biasGrad, DoubleMatrix delta, double aGrad) {
        this.weightGrad = weightGrad;
        this.biasGrad = biasGrad;
        this.delta = delta;
        this.aGrad = aGrad;
    }

    /**
     * Gradients - Constructor for the gradients class (used in StructuredLayer)
     *
     * @param wGrad gradient of the weight matrix
     * @param bGrad gradient of the bias vector
     * @param delt gradient propagated through the layer
     * @param aGrad gradient of the a value
     */
    public Gradients(DoubleMatrix[][] wGrad, DoubleMatrix bGrad, DoubleMatrix[][] delt, double aGrad) {
        this.wGrad = wGrad;
        this.biasGrad = bGrad;
        this.delt = delt;
        this.aGrad = aGrad;
    }

    /**
     * getAGrad - returns the gradient of the a value
     *
     * @return gradient of the a value
     */
    public double getAGrad() {
        return aGrad;
    }

    /**
     * getBiasGrad - returns the gradient of the bias vector
     *
     * @return gradient of the bias vector
     */
    public DoubleMatrix getBiasGrad() {
        return biasGrad;
    }

    /**
     * getDelt - returns the gradient propagated through the layer
     *
     * @return gradient propagated through the layer
     */
    public DoubleMatrix[][] getDelt() {
        return delt;
    }

    /**
     * getDelta - returns the gradient propagated through the layer
     *
     * @return gradient propagated through the layer
     */
    public DoubleMatrix getDelta() {
        return delta;
    }

    /**
     * getWeightGrad - returns the gradient of the weight matrix
     *
     * @return gradient of the weight matrix
     */
    public DoubleMatrix getWeightGrad() {
        return weightGrad;
    }

    /**
     * getWGrad - returns the gradient of the weight matrix
     *
     * @return gradient of the weight matrix
     */
    public DoubleMatrix[][] getWGrad() {
        return wGrad;
    }
}