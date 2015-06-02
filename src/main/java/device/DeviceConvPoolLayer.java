package device;

import Jama.Matrix;

/**
 * Created by jassmanntj on 4/13/2015.
 */
public abstract class DeviceConvPoolLayer {
    public abstract Matrix[] compute(Matrix[] in);
}
