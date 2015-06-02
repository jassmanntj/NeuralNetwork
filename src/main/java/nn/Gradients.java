package nn;

import org.jblas.DoubleMatrix;

public class Gradients {
	public double cost;
	public DoubleMatrix delta;
	public DoubleMatrix thetaGrad;
	public DoubleMatrix biasGrad;
    public DoubleMatrix[][] delt;
    public DoubleMatrix[][] tGrad;
    public double aGrad;
	
	public Gradients(double cost, DoubleMatrix thetaGrad, DoubleMatrix biasGrad, DoubleMatrix delta) {
		this.cost = cost;
		this.thetaGrad = thetaGrad;
		this.biasGrad = biasGrad;
		this.delta = delta;
	}

    public Gradients(double cost, DoubleMatrix thetaGrad, DoubleMatrix biasGrad, DoubleMatrix delta, double aGrad) {
        this.cost = cost;
        this.thetaGrad = thetaGrad;
        this.biasGrad = biasGrad;
        this.delta = delta;
        this.aGrad = aGrad;
    }

    public Gradients(double cost, DoubleMatrix[][] tGrad, DoubleMatrix bGrad, DoubleMatrix[][] delt, double aGrad) {
        this.cost = cost;
        this.tGrad = tGrad;
        this.biasGrad = bGrad;
        this.delt = delt;
        this.aGrad = aGrad;
    }

}