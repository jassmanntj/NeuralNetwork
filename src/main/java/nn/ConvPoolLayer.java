package nn;

import device.DeviceConvPoolLayer;
import org.jblas.DoubleMatrix;

import java.io.BufferedWriter;

/**
 * Created by Tim on 4/1/2015.
 */
public abstract class ConvPoolLayer {
    public abstract DoubleMatrix[][] compute(DoubleMatrix[][] in);
    protected DoubleMatrix[][] gradientCheck(Gradients cr, DoubleMatrix[][] in, DoubleMatrix labels, NeuralNetwork cnn) {
        return null;
    }
    public abstract DoubleMatrix[][] backpropagation(Gradients cr, double momentum, double alpha);
    public abstract Gradients cost(final DoubleMatrix[][] input, final DoubleMatrix[][] output, final DoubleMatrix delta[][]);
    protected double getA() {
        return 0;
    }
    public void pretrain(DoubleMatrix[][] images, int iterations){
    }
    public abstract void writeLayer(BufferedWriter writer);
    protected DoubleMatrix[][] feedForward(DoubleMatrix[][] in) {
        return compute(in);
    };
    public abstract DeviceConvPoolLayer getDevice();
}