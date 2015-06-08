package nn;

import device.DeviceStructuredLayer;
import org.jblas.DoubleMatrix;

import java.io.BufferedWriter;

/**
 * Created by Tim on 4/1/2015.
 */
public abstract class StructuredLayer {
    public abstract DoubleMatrix[][] compute(DoubleMatrix[][] in);
    protected DoubleMatrix[][] gradientCheck(Gradients cr, DoubleMatrix[][] in, DoubleMatrix labels, NeuralNetwork cnn) {
        return null;
    }
    public abstract DoubleMatrix[][] updateWeights(Gradients cr, double momentum, double alpha);
    public abstract Gradients computeGradient(final DoubleMatrix[][] input, final DoubleMatrix[][] output, final DoubleMatrix delta[][]);
    public DoubleMatrix[][] feedForward(DoubleMatrix[][] in) {
        return compute(in);
    };
    public abstract DeviceStructuredLayer getDevice();
    public void initializeParameters() {
    }
}