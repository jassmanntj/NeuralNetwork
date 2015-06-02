package nn;

import Jama.Matrix;
import device.DeviceFCLayer;
import org.jblas.DoubleMatrix;

import java.io.BufferedWriter;
import java.io.IOException;

/**
 * FCLayer - Fully Connected layer
 *
 * @author Tim Jassmann
 * @version 05/26/2015
 */
public class FCLayer {
	private int inputSize;
	private int outputSize;
	private double lambda;
	private DoubleMatrix theta;
	private DoubleMatrix bias;
	private DoubleMatrix biasVelocity;
	private DoubleMatrix thetaVelocity;
    private int activationFunction = Utils.PRELU;
    private double a;
    private double aVelocity;
    private double dropout;
    private DoubleMatrix res;

    /**
     * FCLayer - constructor for FCLayer class
     *
     * Parameters:
     * @param inputSize Size of input to layer
     * @param outputSize Size of output of layer
     * @param lambda weight decay parameter
     * @param dropout percentage of neurons to drop out in training
     * @param activationFunction activation function of layer
     */
	public FCLayer(int inputSize, int outputSize, double lambda, double dropout, int activationFunction) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.lambda = lambda;
        this.dropout = dropout;
        this.activationFunction = activationFunction;
		initializeParams();
	}

    public double getA() {
        return a;
    }

    /**
     * initializeParams - initializes theta, bias, and a values.
     */
	private void initializeParams() {
        double stdev = Math.sqrt(2.0 / ((a * a + 1) * inputSize));
		theta = DoubleMatrix.randn(inputSize, outputSize).muli(stdev);
		thetaVelocity = new DoubleMatrix(inputSize, outputSize);
		biasVelocity = new DoubleMatrix(1, outputSize);
		bias = DoubleMatrix.zeros(1, outputSize);
        a = 0.25;
	}

    /**
     * gradientCheck - gradient checks the layer
     *
     * Parameters:
     * @param input The input to the network
     * @param labels The expected outputs of the network
     * @param gradients The gradients to check
     * @param cnn The network to check the gradients with
     */
	public void gradientCheck(DoubleMatrix[][] input, DoubleMatrix labels, Gradients gradients, NeuralNetwork cnn) {
		double epsilon = 1e-7;

		DoubleMatrix biasG = new DoubleMatrix(bias.rows, bias.columns);
		for(int i = 0; i < bias.length; i++) {
			bias.put(i, bias.get(i)+epsilon);
			Gradients gradientsPlus = cnn.computeCost(input, labels);
			bias.put(i, bias.get(i)-2*epsilon);
			Gradients gradientsMinus = cnn.computeCost(input, labels);
			bias.put(i, bias.get(i)+epsilon);
			biasG.put(i, (gradientsPlus.cost- gradientsMinus.cost)/(2*epsilon));
		}
		DoubleMatrix biasA = biasG.add(gradients.biasGrad);
		DoubleMatrix biasS = biasG.sub(gradients.biasGrad);
		System.out.println("SAE Bias Diff: "+biasS.norm2()/biasA.norm2());

		DoubleMatrix thetaG = new DoubleMatrix(theta.rows, theta.columns);
		for(int i = 0; i < theta.length; i++) {
			theta.put(i, theta.get(i)+epsilon);
			Gradients gradientsPlus = cnn.computeCost(input, labels);
			theta.put(i, theta.get(i)-2*epsilon);
			Gradients gradientsMinus = cnn.computeCost(input, labels);
			theta.put(i, theta.get(i)+epsilon);
			thetaG.put(i, (gradientsPlus.cost- gradientsMinus.cost)/(2*epsilon));
		}
		DoubleMatrix thetaA = thetaG.add(gradients.thetaGrad);
		DoubleMatrix thetaS = thetaG.sub(gradients.thetaGrad);
		System.out.println("SAE Theta Diff: "+thetaS.norm2()/thetaA.norm2());

        a += epsilon;
        Gradients gradientsP = cnn.computeCost(input, labels);
        a -= 2*epsilon;
        Gradients gradientsM = cnn.computeCost(input, labels);
        a += epsilon;
        double aG = (gradientsP.cost- gradientsM.cost)/(2*epsilon);
        System.out.println("SAE a: "+ Math.abs((gradients.aGrad - aG) / (gradients.aGrad + aG)));
	}

    /**
     * cost - compute the gradients and cost of the layer
     * @param input the input to the layer
     * @param output the output of the layer
     * @param delta the gradient propagated to this layer
     *
     * Return:
     * @return gradients of the layer
     */
	public Gradients cost(DoubleMatrix input, DoubleMatrix output, DoubleMatrix delta) {
        double aGrad = Utils.aGrad(activationFunction, res, delta);
		delta.muli(Utils.activationGradient(activationFunction, output, a)).muli(output.ne(0));
		//delta2
		DoubleMatrix delta2 = delta.mmul(theta.transpose());
		//W1grad
		DoubleMatrix thetaGrad = input.transpose().mmul(delta);
		thetaGrad.divi(input.rows * (1 - dropout)).addi(theta.mul(lambda));
		//b1grad
		DoubleMatrix biasGrad = delta.columnSums();
        biasGrad.divi(input.rows * (1 - dropout));
		return new Gradients(0, thetaGrad, biasGrad, delta2, aGrad);
	}

    /**
     * backpropagation - updates weights of the layer
     *
     * Parameters:
     * @param gradients gradients to update weights with
     * @param momentum momentum used in update
     * @param alpha learning rate
     *
     * Return:
     * @return gradient propagated through the layer
     */
	public DoubleMatrix backpropagation(Gradients gradients, double momentum, double alpha) {
		biasVelocity.muli(momentum).add(gradients.biasGrad.mul(alpha));
		thetaVelocity.muli(gradients.thetaGrad.ne(0).mul(momentum)).addi(gradients.thetaGrad.mul(alpha));
        aVelocity = aVelocity * momentum + gradients.aGrad * alpha;
		theta.subi(thetaVelocity.mul(gradients.thetaGrad.ne(0)));
		bias.subi(biasVelocity);
        a -= aVelocity;
        return gradients.delta;
	}

    /**
     * compute - computes the output of the layer (using all layers)
     *
     * Parameters:
     * @param input input to layer
     *
     * Return:
     * @return output of layer
     */
	public DoubleMatrix compute(DoubleMatrix input) {
		DoubleMatrix result = input.mmul(theta);
		result.addiRowVector(bias);
		return Utils.activationFunction(activationFunction, result, a).mul(1 - dropout);
	}

    /**
     * feedforward - computes the output of the layer (with dropout - used for training)
     *
     * Parameters:
     * @param input input to layer
     *
     * Return:
     * @return output of layer
     */
    public DoubleMatrix feedforward(DoubleMatrix input) {
        DoubleMatrix result = input.mmul(theta);
        result.addiRowVector(bias);
        res = result.mul(DoubleMatrix.rand(result.rows, result.columns).ge(dropout));
        return Utils.activationFunction(activationFunction, res, a);
    }

    /*public DoubleMatrix[][] getThetaArr(int patchSize) {
        int channels = theta.rows/(patchSize*patchSize);
        DoubleMatrix[][] res = new DoubleMatrix[theta.columns][channels];
        for(int i = 0; i < theta.columns; i++) {
            for(int j = 0; j < channels; j++) {
                res[i][j] = theta.getRange(patchSize*patchSize*j,patchSize*patchSize*(j+1),i,i+1).reshape(patchSize, patchSize);
            }
        }
        return res;
    }*/

    /**
     * writeLayer - writes the weights of layer to a buffer.
     *
     * Parameters:
     * @param writer the buffer to write to
     */
	public void writeLayer(BufferedWriter writer) {
		try {
            writer.write(activationFunction+","+a+","+true+"\n");
            Utils.printMatrix(theta, writer);
            Utils.printMatrix(bias, writer);
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}

    public DeviceFCLayer getDevice() {
        Matrix t = new Matrix(theta.toArray2());
        Matrix b = new Matrix(bias.toArray2());
        return new DeviceFCLayer(activationFunction, a, t, b, dropout);
    }
}

