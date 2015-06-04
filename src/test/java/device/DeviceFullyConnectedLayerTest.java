package device;

import Jama.Matrix;
import junit.framework.TestCase;
import nn.FCLayer;
import nn.Utils;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class DeviceFullyConnectedLayerTest extends TestCase {
    @Test
    public void testCompute() {
        int input = 100;
        int output = 10;
        double lambda = 5e-5;
        double dropout = 0.5;
        FCLayer fc = new FCLayer(input, output, lambda, dropout, Utils.PRELU);
        DeviceFullyConnectedLayer dfc = fc.getDevice();

        DoubleMatrix in = DoubleMatrix.randn(1,100);
        Matrix din = new Matrix(in.toArray2());

        DoubleMatrix out = fc.compute(in);
        Matrix dout = dfc.compute(din);

        for(int j = 0; j < dout.getRowDimension(); j++) {
            for(int k = 0; k < dout.getColumnDimension(); k++) {
                assertEquals("Output " + j + ":" + k, dout.get(j, k), out.get(j, k), 1e-5);
            }
        }
    }

}