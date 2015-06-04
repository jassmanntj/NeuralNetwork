package nn;

import org.jblas.DoubleMatrix;

public class Gradients {
	public DoubleMatrix delta;
	public DoubleMatrix thetaGrad;
	public DoubleMatrix biasGrad;
    public DoubleMatrix[][] delt;
    public DoubleMatrix[][] tGrad;
    public double aGrad;
	
	public Gradients(DoubleMatrix thetaGrad, DoubleMatrix biasGrad, DoubleMatrix delta) {
		this.thetaGrad = thetaGrad;
		this.biasGrad = biasGrad;
		this.delta = delta;
	}

    public Gradients(DoubleMatrix thetaGrad, DoubleMatrix biasGrad, DoubleMatrix delta, double aGrad) {
        this.thetaGrad = thetaGrad;
        this.biasGrad = biasGrad;
        this.delta = delta;
        this.aGrad = aGrad;
    }

    public Gradients(DoubleMatrix[][] tGrad, DoubleMatrix bGrad, DoubleMatrix[][] delt, double aGrad) {
        this.tGrad = tGrad;
        this.biasGrad = bGrad;
        this.delt = delt;
        this.aGrad = aGrad;
    }

}