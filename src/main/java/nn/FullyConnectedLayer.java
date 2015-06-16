package nn;

import Jama.Matrix;
import device.DeviceFullyConnectedLayer;
import org.jblas.DoubleMatrix;

/**
 * FCLayer - Fully Connected layer
 *
 * @author Tim Jassmann
 * @version 06/12/15
 */
public class FullyConnectedLayer {
	private int inputSize;
	private int outputSize;
	private double lambda;
	private DoubleMatrix weights;
	private DoubleMatrix bias;
	private DoubleMatrix biasVelocity;
	private DoubleMatrix weightVelocity;
    private int activationFunction;
    private double a;
    private double aVelocity;
    private double dropout;

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
	public FullyConnectedLayer(int inputSize, int outputSize, double lambda, double dropout, int activationFunction) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.lambda = lambda;
        this.dropout = dropout;
        this.activationFunction = activationFunction;
		initializeParams();
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
        DoubleMatrix result = input.mmul(weights);
        result.addiRowVector(bias);
        return Utils.activationFunction(activationFunction, result, a).mul(1 - dropout);
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
    public Gradients computeGradient(DoubleMatrix input, DoubleMatrix output, DoubleMatrix delta) {
        DoubleMatrix res = input.mmul(weights).addRowVector(bias);
        double aGrad = Utils.aGradient(activationFunction, res, delta);
        delta.muli(Utils.activationGradient(activationFunction, output, a)).muli(output.ne(0));
        //delta2
        DoubleMatrix delta2 = delta.mmul(weights.transpose());
        //W1grad
        DoubleMatrix thetaGrad = input.transpose().mmul(delta);
        thetaGrad.divi(input.rows * (1 - dropout)).addi(weights.mul(lambda));
        //b1grad
        DoubleMatrix biasGrad = delta.columnSums();
        biasGrad.divi(input.rows * (1 - dropout));
        return new Gradients(thetaGrad, biasGrad, delta2, aGrad);
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
        DoubleMatrix result = input.mmul(weights);
        result.addiRowVector(bias);
        DoubleMatrix res = result.mul(DoubleMatrix.rand(result.rows, result.columns).ge(dropout));
        return Utils.activationFunction(activationFunction, res, a);
    }

    /**
     * getA - returns the a value of the layer
     *
     * @return a
     */
    public double getA() {
        return a;
    }

    /**
     * getBias - returns the bias of the layer
     *
     * @return bias
     */
    public DoubleMatrix getBias() {
        return bias;
    }

    /**
     * getDevice - returns the DeviceFullyConnectedlayer equivalent of this class
     *
     * @return DeviceFullyConnectedlayer equivalent of this class
     */
    public DeviceFullyConnectedLayer getDevice() {
        Matrix t = new Matrix(weights.toArray2());
        Matrix b = new Matrix(bias.toArray2());
        return new DeviceFullyConnectedLayer(t, b, activationFunction, a, dropout);
    }

    /**
     * getWeights - returns the weights of the layer
     *
     * @return weights
     */
    public DoubleMatrix getWeights() {
        return weights;
    }

    /**
     * gradientCheck - gradient checks the layer
     *
     * Parameters:
     * @param input The input to the network
     * @param labels The expected outputs of the network
     * @param gradients The gradients to check
     * @param cnn The network to check the gradients with
     * @param epsilon The value of epsilon for gradient checking
     */
    public double[] gradientCheck(DoubleMatrix[][] input, DoubleMatrix labels, Gradients gradients,
                                  NeuralNetwork cnn, double epsilon) {
        double[] results = new double[3];
        DoubleMatrix biasG = new DoubleMatrix(bias.rows, bias.columns);
        for(int i = 0; i < bias.length; i++) {
            bias.put(i, bias.get(i)+epsilon);
            double gradientsPlus = cnn.computeCost(input, labels);
            bias.put(i, bias.get(i)-2*epsilon);
            double gradientsMinus = cnn.computeCost(input, labels);
            bias.put(i, bias.get(i)+epsilon);
            biasG.put(i, (gradientsPlus- gradientsMinus)/(2*epsilon));
        }
        DoubleMatrix biasA = biasG.add(gradients.getBiasGrad());
        DoubleMatrix biasS = biasG.sub(gradients.getBiasGrad());
        results[1] = biasS.norm2()/biasA.norm2();

        DoubleMatrix thetaG = new DoubleMatrix(weights.rows, weights.columns);
        for(int i = 0; i < weights.length; i++) {
            weights.put(i, weights.get(i)+epsilon);
            double gradientsPlus = cnn.computeCost(input, labels);
            weights.put(i, weights.get(i)-2*epsilon);
            double gradientsMinus = cnn.computeCost(input, labels);
            weights.put(i, weights.get(i)+epsilon);
            thetaG.put(i, (gradientsPlus- gradientsMinus)/(2*epsilon));
        }
        DoubleMatrix thetaA = thetaG.add(gradients.getWeightGrad());
        DoubleMatrix thetaS = thetaG.sub(gradients.getWeightGrad());
        results[0] = thetaS.norm2()/thetaA.norm2();

        if(activationFunction == Utils.PRELU) {
            a += epsilon;
            double gradientsP = cnn.computeCost(input, labels);
            a -= 2 * epsilon;
            double gradientsM = cnn.computeCost(input, labels);
            a += epsilon;
            double aG = (gradientsP - gradientsM) / (2 * epsilon);
            results[1] = Math.abs((gradients.getAGrad() - aG) / (gradients.getAGrad() + aG));
        }
        return results;
    }

    /**
     * initializeParams - initializes theta, bias, and a values.
     */
	public void initializeParams() {
        double stdev = Math.sqrt(2.0 / ((a * a + 1) * inputSize));
        weights = DoubleMatrix.randn(inputSize, outputSize).muli(stdev);
		weightVelocity = new DoubleMatrix(inputSize, outputSize);
		biasVelocity = new DoubleMatrix(1, outputSize);
		bias = DoubleMatrix.zeros(1, outputSize);
        a = 0.25;
	}

    /**
     * updateWeights - updates weights of the layer
     *
     * Parameters:
     * @param gradients gradients to update weights with
     * @param momentum momentum used in update
     * @param alpha learning rate
     *
     * Return:
     * @return gradient propagated through the layer
     */
	public DoubleMatrix updateWeights(Gradients gradients, double momentum, double alpha) {
		biasVelocity.muli(momentum).add(gradients.getBiasGrad().mul(alpha));
		weightVelocity.muli(gradients.getWeightGrad().ne(0).mul(momentum)).addi(gradients.getWeightGrad().mul(alpha));
        aVelocity = aVelocity * momentum + gradients.getAGrad() * alpha;
        weights.subi(weightVelocity.mul(gradients.getWeightGrad().ne(0)));
		bias.subi(biasVelocity);
        a -= aVelocity;
        return gradients.getDelta();
	}
}

