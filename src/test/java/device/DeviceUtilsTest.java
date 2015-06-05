package device;

import Jama.Matrix;
import junit.framework.TestCase;
import nn.Utils;
import org.jblas.DoubleMatrix;
import org.junit.Test;

public class DeviceUtilsTest extends TestCase {

    @Test
    public void testConv2d() throws Exception {
        DoubleMatrix dm1 = DoubleMatrix.randn(4,4);
        DoubleMatrix dm2 = DoubleMatrix.randn(2,2);
        Matrix m1 = new Matrix(dm1.toArray2());
        Matrix m2 = new Matrix(dm2.toArray2());

        DoubleMatrix dout = Utils.conv2d(dm1, dm2, true);
        Matrix mout = DeviceUtils.conv2d(m1, m2);

        for(int i = 0; i < dout.rows; i++) {
            for(int j = 0; j < dout.columns; j++) {
                assertEquals("Conv2d "+i+":"+j, dout.get(i,j), mout.get(i,j), 1e-5);
            }
        }
    }


    @Test
    public void testZCAWhiten() throws Exception {
        int channels = 3;
        double epsilon = 1e-4;
        DoubleMatrix[][] dm = new DoubleMatrix[1][channels];
        Matrix[] m = new Matrix[channels];
        for(int i = 0; i < channels; i++) {
            dm[0][i] = DoubleMatrix.randn(10,10);
            m[i] = new Matrix(dm[0][i].toArray2());
        }
        DoubleMatrix[][] dout = Utils.ZCAWhiten(dm, epsilon);
        Matrix out = DeviceUtils.ZCAWhiten(DeviceUtils.flatten(m), epsilon);

        for(int i = 0; i < dout[0].length; i++) {
            for(int j = 0; j < dout[0][i].rows; j++) {
                for(int k = 0; k < dout[0][i].columns; k++) {
                    assertEquals("Whiten " + i + ":" + j + ":" + k, dout[0][i].get(j, k), out.get(0, i * 10 * 10 + j * 10 + k), 1e-5);
                }
            }
        }
    }

    @Test
    public void testComputeResults() throws Exception {
        DoubleMatrix dRes = DoubleMatrix.rand(1,10);
        Matrix res = new Matrix(dRes.toArray2());

        int[][] dOut = Utils.computeResults(dRes);
        int[] out = DeviceUtils.computeResults(res);

        for(int i = 0; i < dOut[0].length; i++) {
            assertEquals("Res " + i, dOut[0][i], out[i], 1e-5);
        }
    }

    @Test
    public void testFlatten() throws Exception {
        int channels = 3;
        DoubleMatrix[][] dm = new DoubleMatrix[1][channels];
        Matrix[] m = new Matrix[channels];
        for(int i = 0; i < channels; i++) {
            dm[0][i] = DoubleMatrix.randn(10,10);
            m[i] = new Matrix(dm[0][i].toArray2());
        }

        DoubleMatrix dout = Utils.flatten(dm);
        Matrix out = DeviceUtils.flatten(m);
        for(int i = 0; i < dout.rows; i++) {
            for(int j = 0; j < dout.columns; j++) {
                assertEquals("Flatten "+i+":"+j, dout.get(i,j), out.get(i,j), 1e-5);
            }
        }
    }

    @Test
    public void testActivationFunction() throws Exception {
        double a = 0.25;
        DoubleMatrix dm = DoubleMatrix.randn(1,10);
        Matrix m = new Matrix(dm.toArray2());

        DoubleMatrix sigdm = Utils.activationFunction(Utils.SIGMOID, dm, a);
        Matrix sigm = DeviceUtils.activationFunction(DeviceUtils.SIGMOID, m, a);
        DoubleMatrix predm = Utils.activationFunction(Utils.PRELU, dm, a);
        Matrix prem = DeviceUtils.activationFunction(DeviceUtils.PRELU, m, a);
        DoubleMatrix reldm = Utils.activationFunction(Utils.RELU, dm, a);
        Matrix relm = DeviceUtils.activationFunction(DeviceUtils.RELU, m, a);
        DoubleMatrix sofdm = Utils.activationFunction(Utils.SOFTMAX, dm, a);
        Matrix sofm = DeviceUtils.activationFunction(Utils.SOFTMAX, m, a);

        for(int i = 0; i < sigdm.rows; i++) {
            for(int j = 0; j < sigdm.columns; j++) {
                assertEquals("Sigmoid "+i+":"+j, sigdm.get(i,j), sigm.get(i,j), 1e-5);
            }
        }
        for(int i = 0; i < reldm.rows; i++) {
            for(int j = 0; j < reldm.columns; j++) {
                assertEquals("ReLU "+i+":"+j, reldm.get(i,j), relm.get(i,j), 1e-5);
            }
        }
        for(int i = 0; i < predm.rows; i++) {
            for(int j = 0; j < predm.columns; j++) {
                assertEquals("PReLU "+i+":"+j, predm.get(i,j), prem.get(i,j), 1e-5);
            }
        }
        for(int i = 0; i < sofdm.rows; i++) {
            for(int j = 0; j < sofdm.columns; j++) {
                assertEquals("Softmax "+i+":"+j, sofdm.get(i,j), sofm.get(i,j), 1e-5);
            }
        }
    }
}