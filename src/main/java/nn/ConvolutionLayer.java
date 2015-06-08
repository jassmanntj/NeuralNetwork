package nn;

import Jama.Matrix;
import device.DeviceStructuredLayer;
import device.DeviceConvolutionLayer;
import org.jblas.DoubleMatrix;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


/**
 * ConvolutionLayer
 *
 * @author Tim Jassmann
 * @version 05/26/2015
 */
public class ConvolutionLayer extends StructuredLayer {
    private DoubleMatrix weights[][];
    private DoubleMatrix weightVelocity[][];
    private double a;
    private double aVelocity;
    private DoubleMatrix bias;
    private DoubleMatrix biasVelocity;
    private int featureDim;
    private int costFunction;
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
     * @param costFunction cost function to use
     */
    public ConvolutionLayer(int numFeatures, int channels, int featureDim, double lambda, double dropout, int costFunction) {
        this.featureDim = featureDim;
        this.lambda = lambda;
        this.dropout = dropout;
        this.costFunction = costFunction;
        this.numFeatures = numFeatures;
        this.channels = channels;
        initializeParameters();

    }

    public void initializeParameters() {
        bias = new DoubleMatrix(numFeatures);
        biasVelocity = new DoubleMatrix(numFeatures);
        weights = new DoubleMatrix[numFeatures][channels];
        weightVelocity = new DoubleMatrix[numFeatures][channels];
        for(int i = 0; i < numFeatures; i++) {
            for(int j = 0; j < channels; j++) {
                weights[i][j] = initializeTheta(featureDim, channels);
                weightVelocity[i][j] = new DoubleMatrix(featureDim, featureDim);
            }
        }
        if(costFunction == Utils.PRELU) a = .25;
        else a = 0;
    }

    /**
     * pretrain - pretrains the weights of the layer
     * TODO: implementation
     * Parameters:
     * @param input input to layer
     * @param iterations number of iterations of pretraining
     */
    public void pretrain(DoubleMatrix[][] input, int iterations) {
        /*DoubleMatrix patches = ImageLoader.sample(featureDim, featureDim, 10000, input);
        FCLayer ae = new FCLayer(patches.columns, weights.length, 0.035, 3e-3, 5, 1e-3, 0.5);
        ae.train(patches, patches, iterations);
        this.weights = ae.getThetaArr(featureDim);
        this.bias = ae.getBias();
        this.a = ae.getA();*/
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

    /**
     * compute - computes the output of the layer (using all layers)
     *
     * Parameters:
     * @param input input to the layer
     *
     * Return:
     * @return The output of the layer
     */
    public DoubleMatrix[][] compute(final DoubleMatrix[][] input) {
        final DoubleMatrix[][] result = new DoubleMatrix[input.length][weights.length];

        class ConvolutionThread implements Runnable {
            private int imageNum;

            public ConvolutionThread(int imageNum) {
                this.imageNum = imageNum;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < weights.length; feature++) {
                    DoubleMatrix res = new DoubleMatrix(input[imageNum][0].rows-weights[0][0].rows+1, input[imageNum][0].columns-weights[0][0].columns+1);
                    for(int channel = 0; channel < weights[feature].length; channel++) {
                        res.addi(Utils.convolve(input[imageNum][channel], weights[feature][channel], true));
                    }
                    result[imageNum][feature] = Utils.activationFunction(costFunction, res.add(bias.get(feature)), a).mul(1-dropout);
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return result;
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
    public DoubleMatrix[][] feedForward(final DoubleMatrix[][] input) {
        final DoubleMatrix[][] result = new DoubleMatrix[input.length][weights.length];
        z = new DoubleMatrix[input.length][weights.length];
        class ConvolutionThread implements Runnable {
            private int imageNum;

            public ConvolutionThread(int imageNum) {
                this.imageNum = imageNum;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < weights.length; feature++) {
                    DoubleMatrix res = new DoubleMatrix(input[imageNum][0].rows-weights[0][0].rows+1, input[imageNum][0].columns-weights[0][0].columns+1);
                    for(int channel = 0; channel < weights[feature].length; channel++) {
                        res.addi(Utils.convolve(input[imageNum][channel], weights[feature][channel], true));
                    }
                    DoubleMatrix drop = DoubleMatrix.rand(res.rows, res.columns).ge(dropout);
                    z[imageNum][feature] = res.add(bias.get(feature)).mul(drop);
                    result[imageNum][feature] = Utils.activationFunction(costFunction, z[imageNum][feature], a);
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return result;
    }

    /**
     * gradientCheck - performs gradient checking on the layer
     *
     * Parameters:
     * @param gradients gradients of the layer
     * @param in input to the entire network
     * @param labels expected results of the network
     * @param cnn neural network this layer belongs to
     *
     * Return:
     * @return gradient propagated through layer
     */
    protected DoubleMatrix[][] gradientCheck(Gradients gradients, DoubleMatrix[][] in, DoubleMatrix labels, NeuralNetwork cnn) {
        double epsilon = 1e-8;
        this.lambda = 0;
        DoubleMatrix biasG = new DoubleMatrix(bias.length);
        for(int i = 0; i < bias.length; i++) {
            bias.put(i, bias.get(i)+epsilon);
            double gradientsPlus = cnn.computeCost(in, labels);
            bias.put(i, bias.get(i)-2*epsilon);
            double gradientsMinus = cnn.computeCost(in, labels);
            bias.put(i, bias.get(i)+epsilon);
            biasG.put(i, (gradientsPlus- gradientsMinus)/(2*epsilon));
        }
        DoubleMatrix biasA = biasG.add(gradients.biasGrad);
        DoubleMatrix biasS = biasG.sub(gradients.biasGrad);
        System.out.println("CL Bias Diff: " + biasS.norm2() / biasA.norm2());

        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights[i].length; j++) {
                DoubleMatrix thetaG = new DoubleMatrix(gradients.tGrad[i][j].rows, gradients.tGrad[i][j].columns);
                for(int k = 0; k < weights[i][j].length; k++) {
                    weights[i][j].put(k, weights[i][j].get(k)+epsilon);
                    double gradientsPlus = cnn.computeCost(in, labels);
                    weights[i][j].put(k, weights[i][j].get(k)-2*epsilon);
                    double gradientsMinus = cnn.computeCost(in, labels);
                    weights[i][j].put(k, weights[i][j].get(k)+epsilon);
                    thetaG.put(k, (gradientsPlus- gradientsMinus)/(2*epsilon));
                }
                DoubleMatrix thetaA = thetaG.add(gradients.tGrad[i][j]);
                DoubleMatrix thetaS = thetaG.sub(gradients.tGrad[i][j]);
                System.out.println("CL weights "+i+"/"+weights.length+":"+j+"/"+weights[i].length+" Diff: "+thetaS.norm2()/thetaA.norm2());
            }
        }

        a += epsilon;
        double gradientsP = cnn.computeCost(in, labels);
        a -= 2*epsilon;
        double gradientsM = cnn.computeCost(in, labels);
        a += epsilon;
        double aG = (gradientsP- gradientsM)/(2*epsilon);
        System.out.println("CL a: "+ Math.abs((gradients.aGrad - aG) / (gradients.aGrad + aG)));
        return gradients.delt;
    }

    /**
     * cost - computes the gradients of the layer
     *
     * Parameters:
     * @param input input to the layer
     * @param output output of the layer given input
     * @param delta gradient propagated to this layer
     *
     * Return:
     * @return The gradients of the layer
     */
    public Gradients computeGradient(final DoubleMatrix[][] input, final DoubleMatrix[][] output, final DoubleMatrix delta[][]) {
        final DoubleMatrix[][] delt = new DoubleMatrix[input.length][weights[0].length];
        final DoubleMatrix[][] thetaGrad = new DoubleMatrix[weights.length][weights[0].length];
        double aGrad = Utils.aGradient(costFunction, z, delta);
        for(int image = 0; image < input.length; image++) {
            for (int channel = 0; channel < weights[0].length; channel++) {
                delt[image][channel] = new DoubleMatrix(input[0][0].rows, input[0][0].columns);
            }
        }
        for(int feature = 0; feature < weights.length; feature++) {
            for (int channel = 0; channel < weights[0].length; channel++) {
                thetaGrad[feature][channel] = new DoubleMatrix(featureDim, featureDim);
            }
        }
        class ConvolutionThread implements Runnable {
            private int imageNum;

            public ConvolutionThread(int imageNum) {
                this.imageNum = imageNum;
            }

            @Override
            public void run() {
                for(int feature = 0; feature < weights.length; feature++) {
                    delta[imageNum][feature].muli(Utils.activationGradient(costFunction, output[imageNum][feature], a));
                    for(int channel = 0; channel < weights[feature].length; channel++) {
                        delt[imageNum][channel].addi(Utils.convolve(delta[imageNum][feature], Utils.reverseMatrix(weights[feature][channel]), false));
                        thetaGrad[feature][channel].addi(Utils.convolve(input[imageNum][channel], delta[imageNum][feature], true).div(input.length));
                    }
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            Runnable worker = new ConvolutionThread(imageNum);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        for(int i = 0; i < thetaGrad.length; i++) {
            for(int j = 0; j < thetaGrad[i].length; j++) {
                thetaGrad[i][j].addi(weights[i][j].mul(lambda));
            }
        }
        DoubleMatrix bGrad = new DoubleMatrix(bias.length);
        for(int i = 0; i < weights.length; i++) {
            double deltMean = 0;
            for(int j = 0; j < input.length; j++) {
                deltMean += delta[j][i].sum();
            }
            bGrad.put(i, deltMean / input.length);
        }
        //System.out.println(delta[0][0]);

        return new Gradients(thetaGrad, bGrad, delt, aGrad);
    }

    /**
     * backpropagation - updates weights of layer based on backpropagated gradients
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
        biasVelocity.muli(momentum).addi(gradients.biasGrad.mul(alpha));
        bias.subi(biasVelocity);
        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights[i].length; j++) {
                weightVelocity[i][j].muli(momentum).addi(gradients.tGrad[i][j].mul(alpha));
                weights[i][j].subi(weightVelocity[i][j]);
            }
        }
        aVelocity = aVelocity * momentum + gradients.aGrad * alpha;
        a -= aVelocity;
        return gradients.delt;
    }

    public DeviceStructuredLayer getDevice() {
        Matrix b = new Matrix(bias.toArray2());
        Matrix[][] t = new Matrix[weights.length][weights[0].length];
        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights[i].length; j++) {
                t[i][j] = new Matrix(weights[i][j].toArray2());
            }
        }
        return new DeviceConvolutionLayer(t, b, costFunction, a, dropout);
    }
}
