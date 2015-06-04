package device;

import Jama.Matrix;
import junit.framework.TestCase;
import nn.PoolingLayer;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class DevicePoolingLayerTest extends TestCase {
    @Test
    public void testCompute() {
        int poolDim = 2;
        int channels = 3;

        PoolingLayer pl = new PoolingLayer(poolDim, PoolingLayer.MAX);
        DevicePoolingLayer dpl = (DevicePoolingLayer)pl.getDevice();

        DoubleMatrix[][] in = new DoubleMatrix[1][channels];
        Matrix[] din = new Matrix[channels];
        for(int i = 0; i < channels; i++) {
            in[0][i] = DoubleMatrix.randn(8,8);
            din[i] = new Matrix(in[0][i].toArray2());
        }

        DoubleMatrix[][] out = pl.compute(in);
        Matrix[] dout = dpl.compute(din);

        for(int i = 0; i < dout.length; i++) {
            for(int j = 0; j < dout[i].getRowDimension(); j++) {
                for(int k = 0; k < dout[i].getColumnDimension(); k++) {
                    assertEquals("Output "+i+":"+j+":"+k, dout[i].get(j,k), out[0][i].get(j,k), 1e-5);
                }
            }
        }
    }

}