package device;

import Jama.Matrix;

import java.io.Serializable;

/**
 * Created by jassmanntj on 4/13/2015.
 */
public class DeviceConvolutionLayer extends DeviceConvPoolLayer implements Serializable {
    private Matrix theta[][];
    private Matrix bias;
    private int layer1 = DeviceUtils.PRELU;
    private double a;
    private double dropout;

    public DeviceConvolutionLayer(Matrix[][] theta, Matrix bias, int layer1, double a, double dropout) {
        this.theta = theta;
        this.bias = bias;
        this.layer1 = layer1;
        this.a = a;
        this.dropout = dropout;
    }


    public Matrix[] compute(Matrix[] input) {
        Matrix[] result = new Matrix[theta.length];
        for (int feature = 0; feature < theta.length; feature++) {
            Matrix res = new Matrix(input[0].getRowDimension() - theta[0][0].getRowDimension() + 1, input[0].getColumnDimension() - theta[0][0].getColumnDimension() + 1);
            for (int channel = 0; channel < theta[feature].length; channel++) {
                res.plusEquals(DeviceUtils.conv2d(input[channel], theta[feature][channel]));
            }

            result[feature] = DeviceUtils.activationFunction(layer1, res.plus(new Matrix(res.getRowDimension(), res.getColumnDimension(), bias.get(feature, 0))), a).times(1 - dropout);
        }


        return result;
    }

}

