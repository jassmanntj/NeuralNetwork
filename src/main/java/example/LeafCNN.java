package example;

import nn.*;

import java.io.File;

/**
 * LeafCNN - example of cross validation using neural network
 *
 * @author Tim Jassmann
 * @version 06/12/15
 */
public class LeafCNN {

    public static void main(String[] args) throws Exception {
        new LeafCNN().run();
    }

    public void run() throws Exception {
        int patchDim = 3;
        int patchDim2 = 2;
        int poolDim = 2;
        int numFeatures = 16;
        int imageRows = 80;
        int imageColumns = 60;
        double lambda = 5e-5;
        double alpha =  1e-3;
        int channels = 3;
        int hiddenSize = 200;
        int batchSize = 45;
        double momentum = 0.9;
        int iterations = 100;
        double dropout = 0.5;
        int outputSize = 15;
        double convolutionDropout = 0;

        File folder = new File("C:\\Users\\jassmanntj\\Desktop\\CA-Leaves2");
        ImageLoader loader = new ImageLoader(folder, channels, imageColumns, imageRows);

        ConvolutionLayer cl0 = new ConvolutionLayer(numFeatures, channels, patchDim, lambda, convolutionDropout, Utils.PRELU);
        ConvolutionLayer cl1 = new ConvolutionLayer(numFeatures, numFeatures, patchDim2, lambda, convolutionDropout, Utils.PRELU);
        PoolingLayer pl0 = new PoolingLayer(poolDim, PoolingLayer.MAX, poolDim);
        PoolingLayer pl1 = new PoolingLayer(poolDim, PoolingLayer.MAX, poolDim);
        StructuredLayer[] cls = {cl0, pl0, cl1, pl1};

        FullyConnectedLayer sa = new FullyConnectedLayer(numFeatures * 14*19, hiddenSize, lambda, dropout, Utils.PRELU);
        FullyConnectedLayer sa2 = new FullyConnectedLayer(hiddenSize, hiddenSize, lambda, dropout, Utils.PRELU);
        FullyConnectedLayer sc = new FullyConnectedLayer(hiddenSize, outputSize, lambda, 0, Utils.SOFTMAX);
        FullyConnectedLayer[] saes = {sa, sa2, sc};

        NeuralNetwork cnn = new NeuralNetwork(cls, saes, "nnD");
        cnn.crossValidation(loader, 5, iterations, batchSize, momentum, alpha);
    }
}