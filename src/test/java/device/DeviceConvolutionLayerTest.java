package device;

import Jama.Matrix;
import junit.framework.TestCase;
import nn.ConvolutionLayer;
import nn.Utils;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class DeviceConvolutionLayerTest extends TestCase {
    @Test
    public void testCompute() {
        int numFeatures = 5;
        int channels = 3;
        int featureDim = 3;
        double lambda = 5e-5;
        double dropout = 0.5;
        ConvolutionLayer cl = new ConvolutionLayer(numFeatures, channels, featureDim, lambda, dropout, Utils.PRELU);
        DeviceConvolutionLayer dcl = (DeviceConvolutionLayer)cl.getDevice();

        DoubleMatrix[][] in = new DoubleMatrix[1][channels];
        Matrix[] din = new Matrix[channels];
        for(int i = 0; i < channels; i++) {
            in[0][i] = DoubleMatrix.randn(10,10);
            din[i] = new Matrix(in[0][i].toArray2());
        }

        DoubleMatrix[][] out = cl.compute(in);
        Matrix[] dout = dcl.compute(din);

        for(int i = 0; i < dout.length; i++) {
            for(int j = 0; j < dout[i].getRowDimension(); j++) {
                for(int k = 0; k < dout[i].getColumnDimension(); k++) {
                    assertEquals("Output "+i+":"+j+":"+k, dout[i].get(j,k), out[0][i].get(j,k), 1e-5);
                }
            }
        }
    }

}