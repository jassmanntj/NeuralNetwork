package nn;

import Jama.Matrix;
import device.DeviceStructuredLayer;
import device.DeviceConvolutionLayer;
import org.jblas.DoubleMatrix;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


/**
 * ConvolutionLayer
 *
 * @author Tim Jassmann
 * @version 06/12/15
 */
@SuppressWarnings("StatementWithEmptyBody")
public class ConvolutionLayer extends StructuredLayer {
    private DoubleMatrix weights[][];
    private DoubleMatrix weightVelocity[][];
    private double a;
    private double aVelocity;
    private DoubleMatrix bias;
    private DoubleMatrix biasVelocity;
    private int featureDim;
    private int activationFunction;
    private double lambda;
    private final double dropout;
    private DoubleMatrix z[][];
    private int numFeatures;
    private int channels;

    /**
     * ConvolutionLayer - constructor for ConvolutionLayer
     *
     * Parameters:
     * @param numFeatures number of features
     * @param channels number of input channels
     * @param featureDim size of feature
     * @param lambda weight decay
     * @param dropout percentage of output neurons omitted via dropout in training
     * @param activationFunction cost function to use
     */
    public ConvolutionLayer(int numFeatures, int channels, int featureDim, double lambda, double dropout,
                            int activationFunction) {
        this.featureDim = featureDim;
        this.lambda = lambda;
        this.dropout = dropout;
        this.activationFunction = activationFunction;
        this.numFeatures = numFeatures;
        this.channels = channels;
        initializeParameters();
    }

    /**
     * compute - computes the output of the layer (using all layers)
     *
     * Parameters:
     * @param input input to the layer
     *
     * Return:
     * @return The output of the layer
     */
    public DoubleMatrix[][] compute(DoubleMatrix[][] input) {
        DoubleMatrix[][] result = new DoubleMatrix[input.length][weights.length];

        class ConvolutionThread implements Runnable {
            private int imageNum;
            private DoubleMatrix[][] result;
            private DoubleMatrix[][] input;

            public ConvolutionThread(int imageNum, DoubleMatrix[][] input, DoubleMatrix[][] result) {
                this.imageNum = imageNum;
                this.input = input;
                this.result = result;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < weights.length; feature++) {
                    DoubleMatrix res = new DoubleMatrix(input[imageNum][0].rows, input[imageNum][0].columns);
                    for(int channel = 0; channel < weights[feature].length; channel++) {
                        res.addi(Utils.convolve(input[imageNum][channel], weights[feature][channel], res.rows, res.columns));
                    }
                    result[imageNum][feature] = Utils.activationFunction(activationFunction,
                            res.add(bias.get(feature)),
                            a).mul(1 - dropout);
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum, input, result);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return result;
    }

    /**
     * computeGradient - computes the gradients of the layer
     *
     * Parameters:
     * @param input input to the layer
     * @param output output of the layer given input
     * @param delta gradient propagated to this layer
     *
     * Return:
     * @return The gradients of the layer
     */
    public Gradients computeGradient(DoubleMatrix[][] input, DoubleMatrix[][] output, DoubleMatrix delta[][]) {
        final DoubleMatrix[][] delt = new DoubleMatrix[input.length][weights[0].length];
        final DoubleMatrix[][] weightGrad = new DoubleMatrix[weights.length][weights[0].length];
        double aGrad = Utils.aGradient(activationFunction, z, delta);
        for(int image = 0; image < input.length; image++) {
            for (int channel = 0; channel < weights[0].length; channel++) {
                delt[image][channel] = new DoubleMatrix(input[0][0].rows, input[0][0].columns);
            }
        }
        for(int feature = 0; feature < weights.length; feature++) {
            for (int channel = 0; channel < weights[0].length; channel++) {
                weightGrad[feature][channel] = new DoubleMatrix(featureDim, featureDim);
            }
        }
        class ConvolutionThread implements Runnable {
            private int imageNum;
            private DoubleMatrix[][] delta;
            private DoubleMatrix[][] output;
            private DoubleMatrix[][] input;

            public ConvolutionThread(int imageNum, DoubleMatrix[][] delta, DoubleMatrix[][] output,
                                     DoubleMatrix[][] input) {
                this.imageNum = imageNum;
                this.delta = delta;
                this.output = output;
                this.input = input;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < weights.length; feature++) {
                    delta[imageNum][feature].muli(Utils.activationGradient(activationFunction,
                            output[imageNum][feature], a));
                    for(int channel = 0; channel < weights[feature].length; channel++) {
                        delt[imageNum][channel].addi(Utils.convolve(delta[imageNum][feature],
                                Utils.reverseMatrix(weights[feature][channel]), delt[imageNum][channel].rows, delt[imageNum][channel].columns));
                        weightGrad[feature][channel].addi(Utils.convolve(input[imageNum][channel],
                                delta[imageNum][feature], weightGrad[feature][channel].rows, weightGrad[feature][channel].columns).div(input.length));
                    }
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum, delta, output, input);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        for(int row = 0; row < weightGrad.length; row++) {
            for(int column = 0; column < weightGrad[row].length; column++) {
                weightGrad[row][column].addi(weights[row][column].mul(lambda));
            }
        }
        DoubleMatrix bGrad = new DoubleMatrix(bias.length);
        for(int feature = 0; feature < weights.length; feature++) {
            double deltaMean = 0;
            for(int row = 0; row < input.length; row++) {
                deltaMean += delta[row][feature].sum();
            }
            bGrad.put(feature, deltaMean / input.length);
        }
        return new Gradients(weightGrad, bGrad, delt, aGrad);
    }

    /**
     * feedForward - compute the output of the layer (with dropout - used for training)
     *
     * Parameters:
     * @param input input to the layer
     *
     * Return:
     * @return output of layer
     */
    public DoubleMatrix[][] feedForward(DoubleMatrix[][] input) {
        DoubleMatrix[][] result = new DoubleMatrix[input.length][weights.length];
        z = new DoubleMatrix[input.length][weights.length];
        class ConvolutionThread implements Runnable {
            private int imageNum;
            private DoubleMatrix[][] input;
            private DoubleMatrix[][] result;

            public ConvolutionThread(int imageNum, DoubleMatrix[][] input, DoubleMatrix[][] result) {
                this.imageNum = imageNum;
                this.input = input;
                this.result = result;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < weights.length; feature++) {
                    DoubleMatrix res = new DoubleMatrix(input[imageNum][0].rows, input[imageNum][0].columns);
                    for(int channel = 0; channel < weights[feature].length; channel++) {
                        res.addi(Utils.convolve(input[imageNum][channel], weights[feature][channel], res.rows, res.columns));
                    }
                    DoubleMatrix drop = DoubleMatrix.rand(res.rows, res.columns).ge(dropout);
                    z[imageNum][feature] = res.add(bias.get(feature)).mul(drop);
                    result[imageNum][feature] = Utils.activationFunction(activationFunction, z[imageNum][feature], a);
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum, input, result);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return result;
    }

    /**
     * getDevice - returns the DeviceConvolutionLayer equivalent of this class
     *
     * @return DeviceConvolutionLayer equivalent of this class
     */
    public DeviceStructuredLayer getDevice() {
        Matrix biasMatrix = new Matrix(bias.toArray2());
        Matrix[][] weightMatrix = new Matrix[weights.length][weights[0].length];
        for(int feature = 0; feature < weights.length; feature++) {
            for(int channel = 0; channel < weights[feature].length; channel++) {
                weightMatrix[feature][channel] = new Matrix(weights[feature][channel].toArray2());
            }
        }
        return new DeviceConvolutionLayer(weightMatrix, biasMatrix, activationFunction, a, dropout);
    }

    /**
     * gradientCheck - performs gradient checking on the layer
     *
     * Parameters:
     * @param gradients gradients of the layer
     * @param in input to the entire network
     * @param labels expected results of the network
     * @param cnn neural network this layer belongs to
     * @param epsilon epsilon value for gradient checking
     *
     * Return:
     * @return double array containing the norm between numerical and analytical gradients for weights, bias, and a
     */
    public double[] gradientCheck(Gradients gradients, DoubleMatrix[][] in, DoubleMatrix labels,
                                  NeuralNetwork cnn, double epsilon) {
        double[] results = new double[2 + weights.length];
        DoubleMatrix biasG = new DoubleMatrix(bias.length);
        for(int i = 0; i < bias.length; i++) {
            bias.put(i, bias.get(i) + epsilon);
            double gradientsPlus = cnn.computeCost(in, labels);
            bias.put(i, bias.get(i) - 2 * epsilon);
            double gradientsMinus = cnn.computeCost(in, labels);
            bias.put(i, bias.get(i) + epsilon);
            biasG.put(i, (gradientsPlus - gradientsMinus) / (2 * epsilon));
        }
        DoubleMatrix biasA = biasG.add(gradients.getBiasGrad());
        DoubleMatrix biasS = biasG.sub(gradients.getBiasGrad());
        results[weights.length] = biasS.norm2() / biasA.norm2();

        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights[i].length; j++) {
                DoubleMatrix thetaG = new DoubleMatrix(gradients.getWGrad()[i][j].rows,
                        gradients.getWGrad()[i][j].columns);
                for(int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j].put(k, weights[i][j].get(k) + epsilon);
                    double gradientsPlus = cnn.computeCost(in, labels);
                    weights[i][j].put(k, weights[i][j].get(k) - 2 * epsilon);
                    double gradientsMinus = cnn.computeCost(in, labels);
                    weights[i][j].put(k, weights[i][j].get(k) + epsilon);
                    thetaG.put(k, (gradientsPlus - gradientsMinus)/(2 * epsilon));
                }
                DoubleMatrix thetaA = thetaG.add(gradients.getWGrad()[i][j]);
                DoubleMatrix thetaS = thetaG.sub(gradients.getWGrad()[i][j]);
                results[i] = thetaS.norm2() / thetaA.norm2();
            }
        }
        if(activationFunction == Utils.PRELU) {
            a += epsilon;
            double gradientsP = cnn.computeCost(in, labels);
            a -= 2 * epsilon;
            double gradientsM = cnn.computeCost(in, labels);
            a += epsilon;
            double aG = (gradientsP - gradientsM) / (2 * epsilon);
            results[weights.length + 1] = Math.abs((gradients.getAGrad() - aG) / (gradients.getAGrad() + aG));
        }
        return results;
    }

    /**
     * initializeParameters - initializes weights, velocities, and a value.
     */
    public void initializeParameters() {
        bias = new DoubleMatrix(numFeatures);
        biasVelocity = new DoubleMatrix(numFeatures);
        weights = new DoubleMatrix[numFeatures][channels];
        weightVelocity = new DoubleMatrix[numFeatures][channels];
        for(int feature = 0; feature < numFeatures; feature++) {
            for(int channel = 0; channel < channels; channel++) {
                weights[feature][channel] = initializeTheta(featureDim, channels);
                weightVelocity[feature][channel] = new DoubleMatrix(featureDim, featureDim);
            }
        }
        if(activationFunction == Utils.PRELU) a = .25;
        else a = 0;
    }

    /**
     * pretrain - pretrains the weights of the layer
     *
     * Parameters:
     * @param input input to layer
     * @param iterations number of iterations of pretraining
     */
    public void pretrain(DoubleMatrix[][] input, int iterations, double momentum, double alpha) {
        DoubleMatrix patches = Utils.samplePatches(featureDim, featureDim, 10000, input);
        StructuredLayer[] structured = {};
        FullyConnectedLayer fc1 = new FullyConnectedLayer(patches.columns, weights.length, 5e-5, 0.5, Utils.PRELU);
        FullyConnectedLayer fc2 = new FullyConnectedLayer(weights.length, patches.columns, 5e-5, 0.5, Utils.PRELU);
        FullyConnectedLayer[] fcs = {fc1, fc2};
        NeuralNetwork autoencoder = new NeuralNetwork(structured, fcs, "pretrain");
        autoencoder.train(input, Utils.flatten(input), iterations, 100, momentum, alpha, 0);
        this.weights = Utils.expand(fc1.getWeights().transpose(),channels, featureDim, featureDim);
        this.bias = fc1.getBias();
        this.a = fc1.getA();
    }

    /**
     * updateWeights - updates weights of layer based on backpropagated gradients
     *
     * Parameters:
     * @param gradients The gradients of the layer
     * @param momentum The momentum to update the weights with
     * @param alpha The learning rate
     *
     * Return:
     * @return gradient propagated through the layer
     */
    public DoubleMatrix[][] updateWeights(Gradients gradients, double momentum, double alpha) {
        biasVelocity.muli(momentum).addi(gradients.getBiasGrad().mul(alpha));
        bias.subi(biasVelocity);
        for(int feature = 0; feature < weights.length; feature++) {
            for(int channel = 0; channel < weights[feature].length; channel++) {
                weightVelocity[feature][channel].muli(momentum).addi(gradients.getWGrad()[feature][channel].mul(alpha));
                weights[feature][channel].subi(weightVelocity[feature][channel]);
            }
        }
        aVelocity = aVelocity * momentum + gradients.getAGrad() * alpha;
        a -= aVelocity;
        return gradients.getDelt();
    }

    /**
     * initializeTheta - initializes the weights values of a feature of the layer
     * 
     * Parameters:
     * @param featureDim The dimension of the features
     * @param channels The number of input channels
     *
     * Return:
     * @return initialized values for weights
     */
    private DoubleMatrix initializeTheta(int featureDim, int channels) {
        double stdev = Math.sqrt(2.0 / ((a * a + 1) * featureDim * featureDim * channels));
        DoubleMatrix res = DoubleMatrix.randn(featureDim, featureDim);
        res.muli(stdev);
        return res;
    }
}
