package nn;

import Jama.Matrix;
import device.DeviceFullyConnectedLayer;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.io.BufferedWriter;
import java.io.IOException;

public class SoftmaxClassifier extends FCLayer {
	private int inputSize;
	private int outputSize;
	private int m;
	private double lambda;
	private DoubleMatrix theta;
	private DoubleMatrix tVelocity;
	public SoftmaxClassifier(double lambda, int inputSize, int outputSize) {
        super(inputSize, outputSize, lambda, 0, Utils.SOFTMAX);
		this.lambda = lambda;
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		initializeParams();
	}
	
	private void initializeParams() {
        double stdev = Math.sqrt(2.0 / (inputSize));
        theta = DoubleMatrix.randn(inputSize, outputSize).muli(stdev);
		tVelocity = new DoubleMatrix(inputSize, outputSize);
	}

	private DoubleMatrix computeNumericalGradient(DoubleMatrix input, DoubleMatrix output) {
		double epsilon = 0.0001;
		DoubleMatrix numGrad = DoubleMatrix.zeros(theta.rows, theta.columns);
		for(int i = 0; i < theta.rows; i++) {
			for(int j = 0; j < theta.columns; j++) {
				DoubleMatrix thetaPlus = theta.dup();
				DoubleMatrix thetaMinus = theta.dup();
				thetaPlus.put(i,j,thetaPlus.get(i,j)+epsilon);
				thetaMinus.put(i,j,thetaMinus.get(i,j)-epsilon);
				Gradients gradientsPlus = gradcost(input, output, thetaPlus);
				Gradients gradientsMinus = gradcost(input, output, thetaMinus);
				numGrad.put(i,j,(gradientsPlus.cost- gradientsMinus.cost)/(2*epsilon));
			}
		}
		return numGrad;
	}
	
	public Gradients gradcost(DoubleMatrix input, DoubleMatrix output, DoubleMatrix theta) {
		DoubleMatrix res1 = input.mmul(theta);
		DoubleMatrix maxes = res1.rowMaxs();
		DoubleMatrix res = res1.subColumnVector(maxes);
		MatrixFunctions.expi(res);
		res.diviColumnVector(res.rowSums());
		DoubleMatrix thetaGrad = res.sub(output);
		thetaGrad = input.transpose().mmul(thetaGrad);
		thetaGrad.divi(m);
		thetaGrad.addi(theta.mul(lambda));
		MatrixFunctions.logi(res);
		
		double cost = -res.mul(output).sum()/m + theta.mul(theta).sum() * lambda / 2;
		return new Gradients(cost, thetaGrad, null, null);
	}

	public Gradients cost(DoubleMatrix input, DoubleMatrix output, DoubleMatrix delt) {
		m = input.rows;
		DoubleMatrix res = input.mmul(theta);
		DoubleMatrix p = Utils.activationFunction(Utils.SOFTMAX, res, 0);

		DoubleMatrix thetaGrad =input.transpose().mmul(p.sub(output)).div(m).add(theta.mul(lambda));
		DoubleMatrix delta = p.sub(output).mmul(theta.transpose());
		MatrixFunctions.logi(p);
		double cost = -p.mul(output).sum()/m + theta.mul(theta).sum()*lambda/2;
		return new Gradients(cost, thetaGrad, null, delta);
	}

	public DoubleMatrix backpropagation(Gradients c, double momentum, double alpha) {
		tVelocity.muli(momentum).addi(c.thetaGrad.mul(alpha));
		theta.subi(tVelocity);
		return c.delta;
	}

    public void gradientCheck(DoubleMatrix[][] input, DoubleMatrix labels, Gradients gradients, NeuralNetwork cnn){
    }

	public void gradientCheck(DoubleMatrix input, DoubleMatrix output) {
		Gradients result = cost(input, output, theta);
		DoubleMatrix numGrad = computeNumericalGradient(input, output);
		DoubleMatrix gradMin = numGrad.dup();
		DoubleMatrix gradAdd = numGrad.dup();
		gradMin.subi(result.thetaGrad);
		gradAdd.addi(result.thetaGrad);
		System.out.println("SC Diff: " + gradMin.norm2() / gradAdd.norm2());
	}
	
	public void writeLayer(BufferedWriter writer) {
		try {
            writer.write(Utils.SOFTMAX+","+0+","+false+"\n");
            Utils.printMatrix(theta, writer);
			writer.close();
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}

	public DoubleMatrix compute(DoubleMatrix input) {
		return Utils.activationFunction(Utils.SOFTMAX, input.mmul(theta), 0);
	}

    public DoubleMatrix feedforward(DoubleMatrix input) {
        return Utils.activationFunction(Utils.SOFTMAX, input.mmul(theta), 0);
    }

    public DeviceFullyConnectedLayer getDevice() {
        Matrix t = new Matrix(theta.toArray2());
        return new DeviceFullyConnectedLayer(t, null, Utils.SOFTMAX, 0, 0);
    }

}
