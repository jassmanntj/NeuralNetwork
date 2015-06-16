package nn;


import org.jblas.DoubleMatrix;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.Assert;


/**
 * Created by jassmanntj on 6/16/2015.
 */
public class GradientCheckTest {
    static int imageRows = 22;
    static int imageColumns = 22;
    static int numFeatures = 4;
    static int channels = 3;
    static int patchDim = 3;
    static int lambda = 0;
    static int poolDim = 2;
    static int dropout = 0;
    static int hiddenSize = 10;
    static DoubleMatrix[][] data;
    static DoubleMatrix labels;

    @BeforeClass
    public static void init() {
        data = new DoubleMatrix[5][channels];
        for(int i = 0; i < data.length; i++) {
            for(int j = 0; j < data[i].length; j++) {
                data[i][j] = DoubleMatrix.rand(imageRows, imageColumns);
            }
        }
        labels = new DoubleMatrix(5,2);
        labels.put(0,0,1);
        labels.put(1,1,1);
        labels.put(2,0,1);
        labels.put(3,0,1);
        labels.put(4,1,1);
    }

    @Test
    public void testA() {
        ConvolutionLayer cl0 = new ConvolutionLayer(numFeatures, channels, patchDim, lambda, 0, Utils.PRELU);
        PoolingLayer pl0 = new PoolingLayer(poolDim, PoolingLayer.MEAN);
        FullyConnectedLayer sa = new FullyConnectedLayer(numFeatures * 10*10, hiddenSize, lambda, dropout, Utils.PRELU);
        FullyConnectedLayer sa2 = new FullyConnectedLayer(hiddenSize, hiddenSize, lambda, dropout, Utils.PRELU);
        FullyConnectedLayer sc = new FullyConnectedLayer(hiddenSize, 2, lambda, 0, Utils.SOFTMAX);
        StructuredLayer[] cls = {cl0, pl0};
        FullyConnectedLayer[] saes = {sa, sa2, sc};
        NeuralNetwork cnn = new NeuralNetwork(cls, saes, "nnG");
        double[][] result = cnn.gradientCheck(data, labels, 1e-5);
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                Assert.assertEquals("Layer " + i + " Weight " + j, result[i][j], 0, 1e-5);
            }
        }
    }

    @Test
    public void testB() {
        ConvolutionLayer cl0 = new ConvolutionLayer(numFeatures, channels, patchDim, lambda, 0, Utils.PRELU);
        PoolingLayer pl0 = new PoolingLayer(poolDim, PoolingLayer.MEAN);
        FullyConnectedLayer sa = new FullyConnectedLayer(numFeatures * 10*10, hiddenSize, lambda, dropout, Utils.PRELU);
        FullyConnectedLayer sa2 = new FullyConnectedLayer(hiddenSize, hiddenSize, lambda, dropout, Utils.PRELU);
        FullyConnectedLayer sc = new FullyConnectedLayer(hiddenSize, 2, lambda, 0, Utils.SOFTMAX);
        StructuredLayer[] cls = {cl0, pl0};
        FullyConnectedLayer[] saes = {sa, sa2, sc};
        NeuralNetwork cnn = new NeuralNetwork(cls, saes, "nnG");
        double[][] result = cnn.gradientCheck(data, labels, 1e-5);
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                Assert.assertEquals("Layer " + i + " Weight " + j, result[i][j], 0, 1e-5);
            }
        }
    }

    @Test
    public void testC() {
        ConvolutionLayer cl0 = new ConvolutionLayer(numFeatures, channels, patchDim, lambda, 0, Utils.RELU);
        PoolingLayer pl0 = new PoolingLayer(poolDim, PoolingLayer.MEAN);
        FullyConnectedLayer sa = new FullyConnectedLayer(numFeatures * 10*10, hiddenSize, lambda, dropout, Utils.RELU);
        FullyConnectedLayer sa2 = new FullyConnectedLayer(hiddenSize, hiddenSize, lambda, dropout, Utils.RELU);
        FullyConnectedLayer sc = new FullyConnectedLayer(hiddenSize, 2, lambda, 0, Utils.SOFTMAX);
        StructuredLayer[] cls = {cl0, pl0};
        FullyConnectedLayer[] saes = {sa, sa2, sc};
        NeuralNetwork cnn = new NeuralNetwork(cls, saes, "nnG");
        double[][] result = cnn.gradientCheck(data, labels, 1e-5);
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                Assert.assertEquals("Layer " + i + " Weight " + j, result[i][j], 0, 1e-5);
            }
        }
    }

    @Test
    public void testD() {
        ConvolutionLayer cl0 = new ConvolutionLayer(numFeatures, channels, patchDim, lambda, 0, Utils.SIGMOID);
        PoolingLayer pl0 = new PoolingLayer(poolDim, PoolingLayer.MEAN);

        FullyConnectedLayer sa = new FullyConnectedLayer(numFeatures * 10*10, hiddenSize, lambda, dropout, Utils.SIGMOID);
        FullyConnectedLayer sa2 = new FullyConnectedLayer(hiddenSize, hiddenSize, lambda, dropout, Utils.SIGMOID);
        FullyConnectedLayer sc = new FullyConnectedLayer(hiddenSize, 2, lambda, 0, Utils.SOFTMAX);
        StructuredLayer[] cls = {cl0, pl0};
        FullyConnectedLayer[] saes = {sa, sa2, sc};
        NeuralNetwork cnn = new NeuralNetwork(cls, saes, "nnG");
        double[][] result = cnn.gradientCheck(data, labels, 1e-5);
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                Assert.assertEquals("Layer " + i + " Weight " + j, result[i][j], 0, 1e-5);
            }
        }
    }

    @Test
    public void testE() {
        ConvolutionLayer cl0 = new ConvolutionLayer(numFeatures, channels, patchDim, lambda, 0, Utils.NONE);
        PoolingLayer pl0 = new PoolingLayer(poolDim, PoolingLayer.MEAN);

        FullyConnectedLayer sa = new FullyConnectedLayer(numFeatures * 10*10, hiddenSize, lambda, dropout, Utils.NONE);
        FullyConnectedLayer sa2 = new FullyConnectedLayer(hiddenSize, hiddenSize, lambda, dropout, Utils.NONE);
        FullyConnectedLayer sc = new FullyConnectedLayer(hiddenSize, 2, lambda, 0, Utils.SOFTMAX);
        StructuredLayer[] cls = {cl0, pl0};
        FullyConnectedLayer[] saes = {sa, sa2, sc};
        NeuralNetwork cnn = new NeuralNetwork(cls, saes, "nnG");
        double[][] result = cnn.gradientCheck(data, labels, 1e-5);
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                Assert.assertEquals("Layer " + i + " Weight " + j, result[i][j], 0, 1e-5);
            }
        }
    }

}
