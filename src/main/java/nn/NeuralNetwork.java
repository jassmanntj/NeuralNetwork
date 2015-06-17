package nn;

import device.DeviceStructuredLayer;
import device.DeviceFullyConnectedLayer;
import device.DeviceNeuralNetwork;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.Random;

/**
 * NeuralNetwork - Encapsulates layers for a neural network
 *
 * @author Timothy Jassmann
 * @version 06/16/2015
 */
public class NeuralNetwork {
    private StructuredLayer[] structuredLayers;
    private FullyConnectedLayer[] fullyConnectedLayers;
    private String name;
    private Random r;
    private double cost;
    private double previousCost;

    /**
     * NeuralNetwork - constructor for neural network class
     *
     * @param structuredLayers array of structured layers for neural network
     * @param fullyConnectedLayers array of fully connected layers for neural network
     * @param name name to save neural network results as
     */
    public NeuralNetwork(StructuredLayer[] structuredLayers, FullyConnectedLayer[] fullyConnectedLayers, String name) {
        this.structuredLayers = structuredLayers;
        this.fullyConnectedLayers = fullyConnectedLayers;
        this.name = name;
        this.cost = Double.MAX_VALUE;
        this.previousCost = Double.MAX_VALUE;
        r = new Random(System.currentTimeMillis());
    }

    /**
     * train - trains a network on input data and labels with stochastic gradient descent
     * @param data input data to train on
     * @param labels input labels to train on
     * @param testData data to test network with while training
     * @param testLabels labels to test network with while training
     * @param iterations iterations to train for
     * @param batchSize batch size to train with
     * @param momentum momentum to train with
     * @param alpha learning rate to train with
     * @param set string appended onto neural network name identifying network when written
     */
    public void train(DoubleMatrix[][] data, DoubleMatrix labels, DoubleMatrix[][] testData, DoubleMatrix testLabels,
                      int iterations, int batchSize, double momentum, double alpha, int set) {
        int[] indices = new int[data.length];
        for(int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        for(int iteration = 0; iteration < iterations; iteration++) {
            System.out.println("Alpha: " + alpha);
            shuffleInts(indices, indices.length);
            for(int batch = 0; batch < data.length / batchSize; batch++) {
                System.out.print(batch + " ");
                DoubleMatrix[][] batchData = new DoubleMatrix[batchSize][data[0].length];
                DoubleMatrix batchLabels = new DoubleMatrix(batchSize, labels.columns);
                for(int image = 0; image < batchData.length; image++) {
                    for(int channel = 0; channel < data[image].length; channel++) {
                        batchData[image][channel] = data[indices[batch * batchSize + image]][channel].dup();
                    }
                    batchLabels.putRow(image, labels.getRow(indices[batch * batchSize + image]));
                }
                Utils.alterImages(batchData);

                DoubleMatrix[][][] convResults = new DoubleMatrix[structuredLayers.length + 1][][];
                convResults[0] = batchData;
                for(int k = 0; k < structuredLayers.length; k++) {
                    convResults[k + 1] = structuredLayers[k].feedForward(convResults[k]);
                }
                DoubleMatrix[] fcResults = new DoubleMatrix[fullyConnectedLayers.length + 1];
                fcResults[0] = Utils.flatten(convResults[structuredLayers.length]);
                for(int k = 0; k < fullyConnectedLayers.length; k++) {
                    fcResults[k + 1] = fullyConnectedLayers[k].feedforward(fcResults[k]);
                }
                DoubleMatrix delta = fcResults[fcResults.length - 1].sub(batchLabels);
                for(int k = fullyConnectedLayers.length - 1; k >= 0; k--) {
                    Gradients cr = fullyConnectedLayers[k].computeGradient(fcResults[k], fcResults[k + 1], delta);
                    delta = fullyConnectedLayers[k].updateWeights(cr, momentum, alpha);

                }
                Gradients cr;
                DoubleMatrix[][] delt = Utils.expand(delta, convResults[structuredLayers.length][0].length,
                        convResults[structuredLayers.length][0][0].rows,
                        convResults[structuredLayers.length][0][0].columns);
                for(int k = structuredLayers.length-1; k >= 0; k--) {
                    cr = structuredLayers[k].computeGradient(convResults[k], convResults[k + 1], delt);
                    delt = structuredLayers[k].updateWeights(cr, momentum, alpha);
                }
            }
            System.out.println("\nSet: "+set+"\nIteration "+iteration);

            System.out.println("Train");
            DoubleMatrix res = compute(data, batchSize, labels);
            if(cost > previousCost) alpha *= 0.75;
            previousCost = cost;
            compareResults(Utils.computeRanking(res), labels);
            if(testData != null) {
                System.out.println("Test");
                DoubleMatrix testRes = compute(testData, batchSize, testLabels);
                compareResults(Utils.computeRanking(testRes), testLabels);
            }

        }
        writeDevice(name+set);
    }

    /**
     * train - trains a network on input data and labels with stochastic gradient descent
     * @param data input data to train on
     * @param labels input labels to train on
     * @param iterations iterations to train for
     * @param batchSize batch size to train with
     * @param momentum momentum to train with
     * @param alpha learning rate to train with
     * @param set string appended onto neural network name identifying network when written
     */
    public void train(DoubleMatrix[][] data, DoubleMatrix labels, int iterations, int batchSize,
                      double momentum, double alpha, int set) {
        train(data, labels, null, null, iterations, batchSize, momentum, alpha, set);
    }

    /**
     * crossValidation - performs k-fold cross validation
     *
     * @param loader loader to get the data from
     * @param k the number of partitions of the data for validation
     * @param iterations the iterations to train for
     * @param batchSize the batchSize to train the networks with
     * @param momentum the momentum to train the networks with
     * @param alpha the learning rate to train the networks with
     */
    public void crossValidation(Loader loader, int k, int iterations, int batchSize, double momentum, double alpha) {
        for(int i = 0; i < k; i++) {
            DoubleMatrix[][] trainImages = loader.getTrainData(i,k);
            DoubleMatrix trainLabels = loader.getTrainLabels(i,k);
            DoubleMatrix[][] testImages = loader.getTestData(i,k);
            DoubleMatrix testLabels = loader.getTestLabels(i,k);
            train(trainImages, trainLabels, testImages, testLabels, iterations, batchSize, momentum, alpha, i);
            compareClasses(Utils.computeRanking(compute(testImages, batchSize)), testLabels, loader.getLabelMap());
            resetWeights();
        }
    }

    /**
     * resetWeights - resets all weights of the network to randomized values.
     */
    public void resetWeights() {
        previousCost = Double.MAX_VALUE;
        cost = Double.MAX_VALUE;
        for (StructuredLayer structuredLayer : structuredLayers) {
            structuredLayer.initializeParameters();
        }
        for (FullyConnectedLayer fullyConnectedLayer : fullyConnectedLayers) {
            fullyConnectedLayer.initializeParams();
        }
    }

    /**
     * compareClasses - prints the accuracy of the network with respect to each classification
     *
     * @param result result of network
     * @param labels labels of network
     * @param labelMap hashmap that maps string labels to numerical labels
     */
    public void compareClasses(int[][] result, DoubleMatrix labels, HashMap<String, Double> labelMap) {
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
                }
            }
            if(labelNo == result[i][0]) {
                count[labelNo]++;
            }
            totalCount[labelNo]++;
        }
        for(int i = 0; i < count.length; i++) {
            System.out.println(newMap.get((double) i) + ": " + count[i] + "/" + totalCount[i]
                    + " = " + (count[i] / totalCount[i]));
        }
    }

    /**
     * reverseMap - reverses a hashmap such that keys become values and values become keys
     *
     * @param map hashmap to reverse
     *
     * @return reversed hashmap
     */
    private HashMap<Double, String> reverseMap(HashMap<String, Double> map) {
        HashMap<Double, String> newMap = new HashMap<Double, String>();
        for(String key : map.keySet()) {
            newMap.put(map.get(key), key);
        }
        return newMap;
    }

    /**
     * compareResults - compares results of network to expected values. Prints the number
     *                  of results that had the correct result ranked first, second, third, and so on.
     *
     * @param result Result of network
     * @param labels Labels corresponding to results of network
     *
     * @return true if all results are the correct classification
     */
    public boolean compareResults(int[][] result, DoubleMatrix labels) {
        double[] sums = new double[result[0].length];
        System.out.println(result.length+":"+labels.rows+":"+labels.columns);
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                sums[j] += labels.get(i, result[i][j]);
            }
        }
        for(int i = 0; i < sums.length; i++) {
            if(sums[i] > 0) System.out.println(i + ": " + sums[i] + "/" + result.length
                                                 + " = " + (sums[i] / result.length));
        }
        return(sums[0] == result.length);
    }

    /**
     * computeCost - returns the cost of the network
     *
     * @param input input to network
     * @param labels labels corresponding to input
     *
     * @return cost of the network
     */
    public double computeCost(DoubleMatrix[][] input, DoubleMatrix labels) {
        DoubleMatrix[][][] convResults = new DoubleMatrix[structuredLayers.length + 1][][];
        convResults[0] = input;
        for(int k = 0; k < structuredLayers.length; k++) {
            convResults[k + 1] = structuredLayers[k].compute(convResults[k]);
        }
        DoubleMatrix[] ldsResults = new DoubleMatrix[fullyConnectedLayers.length + 1];
        ldsResults[0] = Utils.flatten(convResults[structuredLayers.length]);
        for(int k = 0; k < fullyConnectedLayers.length; k++) {
            ldsResults[k + 1] = fullyConnectedLayers[k].compute(ldsResults[k]);
        }
        DoubleMatrix p = MatrixFunctions.log(ldsResults[ldsResults.length - 1]);
        return -p.mul(labels).sum() / input.length;
    }

    /**
     * gradientCheck - performs gradient checking on the neural network
     *
     * @param input input to perform gradient checking with
     * @param labels labels to perform gradient checking with
     * @param epsilon epsilon value for gradient checking
     *
     * @return double array containing the norm between numerical and analytical gradients for weights, bias, and a
     */
    public double[][] gradientCheck(DoubleMatrix[][] input, DoubleMatrix labels, double epsilon) {
        DoubleMatrix[][][] convResults = new DoubleMatrix[structuredLayers.length + 1][][];
        convResults[0] = input;
        double[][] results = new double[fullyConnectedLayers.length + structuredLayers.length][];
        for(int i = 0; i < structuredLayers.length; i++) {
            convResults[i + 1] = structuredLayers[i].feedForward(convResults[i]);
        }
        DoubleMatrix[] ldsResults = new DoubleMatrix[fullyConnectedLayers.length+1];
        ldsResults[0] = Utils.flatten(convResults[structuredLayers.length]);
        for(int i = 0; i < fullyConnectedLayers.length; i++) {
            ldsResults[i + 1] = fullyConnectedLayers[i].feedforward(ldsResults[i]);
        }
        Gradients cr;
        DoubleMatrix delta = ldsResults[ldsResults.length-1].sub(labels);
        for(int k = fullyConnectedLayers.length - 1; k >= 0; k--) {
            cr = fullyConnectedLayers[k].computeGradient(ldsResults[k], ldsResults[k + 1], delta);
            delta = cr.getDelta();
            results[structuredLayers.length + k] =
                    fullyConnectedLayers[k].gradientCheck(input, labels, cr, this, epsilon);
        }
        DoubleMatrix[][] delt = Utils.expand(delta, convResults[structuredLayers.length][0].length,
                convResults[structuredLayers.length][0][0].rows, convResults[structuredLayers.length][0][0].columns);
        for(int k = structuredLayers.length - 1; k >= 0; k--) {
            cr = structuredLayers[k].computeGradient(convResults[k], convResults[k + 1], delt);
            results[k] = structuredLayers[k].gradientCheck(cr, input, labels, this, epsilon);
            delt = cr.getDelt();
        }
        return results;
    }

    /**
     * shuffleInts - randomly shuffles an array of ints
     *
     * @param integers integers to shuffle
     * @param iterations iterations to shuffle integers for
     */
    private void shuffleInts(int[] integers, int iterations) {
        for(int i = 0; i < iterations; i++) {
            int a = r.nextInt(integers.length);
            int b = r.nextInt(integers.length);
            int temp = integers[a];
            integers[a] = integers[b];
            integers[b] = temp;
        }
    }

    /**
     * compute - computes the output of the network for the input
     *
     * @param input input matrix to compute
     * @param batchSize batch size for computation
     *
     * @return output of the neural network
     */
    public DoubleMatrix compute(DoubleMatrix[][] input, int batchSize) {
        DoubleMatrix res = null;
        for(int j = 0; j < input.length/batchSize; j++) {
            DoubleMatrix[][] in = new DoubleMatrix[batchSize][];
            System.arraycopy(input, j * batchSize, in, 0, batchSize);
            for (StructuredLayer structuredLayer : structuredLayers) {
                in = structuredLayer.compute(in);
            }
            DoubleMatrix fin = Utils.flatten(in);
            for (FullyConnectedLayer fullyConnectedLayer : fullyConnectedLayers) {
                fin = fullyConnectedLayer.compute(fin);
            }
            if (res == null) res = fin;
            else res = DoubleMatrix.concatVertically(res, fin);
        }
        return res;
    }

    /**
     * compute - computes the output of the network for the input and prints the cost
     *
     * @param input input matrix to compute
     * @param batchSize batch size for computation
     * @param labels labels to compare output to for cost
     *
     * @return output of the neural network
     */
    public DoubleMatrix compute(DoubleMatrix[][] input, int batchSize, DoubleMatrix labels) {
        DoubleMatrix res = null;
        cost = 0;
        for(int j = 0; j < input.length / batchSize; j++) {
            DoubleMatrix[][] in = new DoubleMatrix[batchSize][];
            DoubleMatrix labs = labels.getRange(j * batchSize, j * batchSize + batchSize, 0, labels.columns);
            System.arraycopy(input, j * batchSize, in, 0, batchSize);
            for (StructuredLayer structuredLayer : structuredLayers) {
                in = structuredLayer.compute(in);
            }
            DoubleMatrix fin = Utils.flatten(in);
            for (int i = 0; i < fullyConnectedLayers.length-1; i++) {
                fin = fullyConnectedLayers[i].compute(fin);
            }
            DoubleMatrix out = fullyConnectedLayers[fullyConnectedLayers.length - 1].compute(fin);
            if (res == null) res = out;
            else res = DoubleMatrix.concatVertically(res, out);
            DoubleMatrix p = MatrixFunctions.log(out);
            cost += -p.mul(labs).sum() / (out.rows * (input.length / batchSize));
        }
        System.out.println("Cost: " + cost);
        return res;
    }

    /**
     * writeDevice - writes the device neural network to a file which can be loaded by
     *               a mobile application
     * @param filename Filename to save file to
     */
    public void writeDevice(String filename) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
            DeviceStructuredLayer[] cp = new DeviceStructuredLayer[structuredLayers.length];
            DeviceFullyConnectedLayer[] fc = new DeviceFullyConnectedLayer[fullyConnectedLayers.length];
            for(int i = 0; i < structuredLayers.length; i++) {
                cp[i] = structuredLayers[i].getDevice();
            }
            for(int i = 0; i < fullyConnectedLayers.length; i++) {
                fc[i] = fullyConnectedLayers[i].getDevice();
            }
            DeviceNeuralNetwork nn = new DeviceNeuralNetwork(cp, fc);
            out.writeObject(nn);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
