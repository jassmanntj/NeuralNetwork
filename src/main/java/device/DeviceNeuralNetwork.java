package device;

import Jama.Matrix;

import java.io.IOException;
import java.io.Serializable;

/**
 * Created by jassmanntj on 4/13/2015.
 */

public class DeviceNeuralNetwork implements Serializable{
    DeviceConvPoolLayer[] cls;
    DeviceFCLayer[] fcs;

    public DeviceNeuralNetwork(DeviceConvPoolLayer[] cls, DeviceFCLayer[] fcs) throws IOException {
        this.cls = cls;
        this.fcs = fcs;
    }

    public Matrix compute(Matrix[] input) {
        Matrix in = null;
        for(int i = 0; i < cls.length; i++) {
            input = cls[i].compute(input);
        }
        in = DeviceUtils.flatten(input);
        for(int i = 0; i < fcs.length; i++) {
            in = fcs[i].compute(in);
        }
        return in;
    }

    private String getString(Matrix mat) {
        String s = "";
        for(int i = 0; i < mat.getRowDimension(); i++) {
            for(int j = 0; j < mat.getColumnDimension(); j++) {
                s += mat.get(i,j);
            }
        }
        return s;
    }
}
