package device;

import Jama.Matrix;

import java.io.Serializable;

/**
 * Created by jassmanntj on 4/13/2015.
 */
public class DeviceFCLayer implements Serializable {
    private int activation;
    private double a;
    private Matrix theta;
    private Matrix bias;
    private double dropout;

    public DeviceFCLayer(int activation, double a, Matrix theta, Matrix bias, double dropout) {
        this.activation = activation;
        this.a = a;
        this.theta = theta;
        this.bias = bias;
        this.dropout = dropout;
    }

    public Matrix compute(Matrix input) {
        Matrix result = input.times(theta);
        if(bias != null) result.plusEquals(bias);
        return DeviceUtils.activationFunction(activation, result, a).times(1 - dropout);
    }
}
