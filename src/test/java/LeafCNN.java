import nn.*;
import org.jblas.DoubleMatrix;

import java.io.File;
import java.util.HashMap;

/**
 * Created by Tim on 4/1/2015.
 */
public class LeafCNN {

    public static void main(String[] args) throws Exception {
        new LeafCNN().run();
    }

    public void run() throws Exception {
        DoubleMatrix[][] images;
        DoubleMatrix labels;
        int patchDim = 3;
        int patchDim2 = 2;
        int poolDim = 2;
        int numFeatures = 16;
        //int hiddenSize = 200;
        int imageRows = 80;
        int imageColumns = 60;
        double lambda = 5e-5;
        double alpha =  1e-3;
        int channels = 3;
        int hiddenSize = 200;
        int hiddenSize2 = 200;
        int batchSizeLoad = 90;//42
        int batchSize = 45;
        double momentum = 0.9;
        int iterations = 30;
        double dropout = 0.5;

        String resFile = "Testing";

        ImageLoader loader = new ImageLoader();
        File folder = new File("C:\\Users\\jassmanntj\\Desktop\\CA-Leaves2");
        HashMap<String, Double> labelMap = loader.getLabelMap(folder);
        loader.loadFolder(folder, channels, imageColumns, imageRows, labelMap);

        for(int i = 0; i < 1; i++) {
            images = loader.getImgArr(i, false, false, batchSizeLoad);
            labels = loader.getLabels(i, false, false, batchSizeLoad);
            DoubleMatrix[][] testImages = loader.getTestArr(i, false, false, batchSizeLoad);
            DoubleMatrix testLabels = loader.getTestLabels(i, false, false, batchSizeLoad);
            StructuredLayer[] cls = new StructuredLayer[2];
            cls[0] = new ConvolutionLayer(numFeatures, channels, patchDim, lambda, 0, Utils.PRELU);
            //cls[2] = new ConvolutionLayer(numFeatures, numFeatures, patchDim2, lambda, dropout, Utils.PRELU);
            //cls[4] = new ConvolutionLayer(numFeatures, numFeatures, patchDim2, lambda, 0, Utils.PRELU);
            cls[1] = new PoolingLayer(poolDim, PoolingLayer.MAX);
            //cls[3] = new PoolingLayer(poolDim, PoolingLayer.MAX);

            FCLayer sa = new FCLayer(numFeatures * 29*39, hiddenSize, lambda, dropout, Utils.PRELU);
            //FCLayer sa2 = new FCLayer(hiddenSize, hiddenSize, lambda, dropout, Utils.PRELU);

            FCLayer sc = new FCLayer(hiddenSize, labels.columns, lambda, 0, Utils.SOFTMAX);
            FCLayer[] saes = {sa, sc}; //{sa, sa2};
            NeuralNetwork cnn = new NeuralNetwork(cls, saes, "TestNNz"+i);
            cnn.train(images, labels, testImages, testLabels, iterations, batchSize, momentum, alpha, i);
            compareClasses(Utils.computeResults(cnn.compute(testImages, batchSize)), testLabels, labelMap);
        }
    }

    public void compareClasses(int[][] result, DoubleMatrix labels, HashMap<String, Double> labelMap) throws Exception {
        HashMap<Double, String> newMap = reverseMap(labelMap);
        double[] count = new double[labels.columns];
        double[] totalCount = new double[labels.columns];
        for(int i = 0; i < result.length; i++) {
            int labelNo = -1;
            for(int j = 0; j < labels.columns; j++) {
                if((int)labels.get(i, j) == 1) {
                    if(labelNo == -1) {
                        labelNo = j;
                    }
                    else throw new Exception("Invalid Labels");
                }
            }
            if(labelNo == result[i][0]) {
                count[labelNo]++;
            }
            totalCount[labelNo]++;
        }
        for(int i = 0; i < count.length; i++) {
            System.out.println(newMap.get((double)i)+": " +count[i]+"/"+totalCount[i]+" = "+(count[i]/totalCount[i]));
        }
    }

    private HashMap<Double, String> reverseMap(HashMap<String, Double> map) {
        HashMap<Double, String> newMap = new HashMap<Double, String>();
        for(String key : map.keySet()) {
            newMap.put(map.get(key), key);
        }
        return newMap;
    }

}