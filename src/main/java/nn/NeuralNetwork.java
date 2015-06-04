package nn;

import device.DeviceStructuredLayer;
import device.DeviceFullyConnectedLayer;
import device.DeviceNeuralNetwork;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;

/**
 * Created by Tim on 3/29/2015.
 */
public class NeuralNetwork {
    private StructuredLayer[] cls;
    private FCLayer[] lds;
    private String name;
    private Random r;
    private double cost;
    private double previousCost;
    private static final boolean DEBUG = false;

    public NeuralNetwork(StructuredLayer[] cls, FCLayer[] lds, String name) {
        this.cls = cls;
        this.lds = lds;
        this.name = name;
        this.cost = 1e9;
        r = new Random(System.currentTimeMillis());
    }

    public void train(DoubleMatrix[][] input, DoubleMatrix labels, DoubleMatrix[][] test, DoubleMatrix testLab, int iterations, int batchSize, double momentum, double alpha, double features) throws IOException {
        int[] indices = new int[input.length];
        for(int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        for(int i = 0; i < iterations; i++) {
            previousCost = cost;
            System.out.println("Alpha: "+alpha);
            shuffleIndices(indices, indices.length);
            boolean br = false;
            for(int j = 0; j < input.length/batchSize; j++) {
                System.out.print(j + " ");
                DoubleMatrix[][] in = new DoubleMatrix[batchSize][input[0].length];
                DoubleMatrix labs = new DoubleMatrix(batchSize, labels.columns);
                for(int k = 0; k < in.length; k++) {
                    for(int l = 0; l < input[k].length; l++) {
                        in[k][l] = input[indices[j * batchSize + k]][l].dup();
                    }
                    labs.putRow(k, labels.getRow(indices[j*batchSize+k]));
                }
                Utils.alter(in);

                if(DEBUG) gradientCheck(in, labs);
                DoubleMatrix[][][] convResults = new DoubleMatrix[cls.length+1][][];
                convResults[0] = in;
                for(int k = 0; k < cls.length; k++) {
                    convResults[k+1] = cls[k].feedForward(convResults[k]);
                }
                DoubleMatrix[] ldsResults = new DoubleMatrix[lds.length+1];
                ldsResults[0] = Utils.flatten(convResults[cls.length]);
                for(int k = 0; k < lds.length; k++) {
                    ldsResults[k+1] = lds[k].feedforward(ldsResults[k]);
                }
                DoubleMatrix delta = ldsResults[ldsResults.length-1].sub(labs);
                for(int k = lds.length-1; k >= 0; k--) {
                    Gradients cr = lds[k].cost(ldsResults[k], ldsResults[k+1], delta, labs);
                    delta = lds[k].backpropagation(cr, momentum, alpha);
                    if(Double.isNaN(lds[k].getA())) {
                        System.out.println("FC"+k);
                        br = true;
                    }

                }
                Gradients cr;
                DoubleMatrix[][] delt = Utils.expand(delta, convResults[cls.length][0].length, convResults[cls.length][0][0].rows, convResults[cls.length][0][0].columns);
                for(int k = cls.length-1; k >= 0; k--) {
                    cr = cls[k].cost(convResults[k], convResults[k + 1], delt);
                    delt = cls[k].backpropagation(cr, momentum, alpha);
                    if(Double.isNaN(cls[k].getA())) {
                        System.out.println("CLS"+k);
                        br = true;
                    }
                }
                if(br) break;
            }
            System.out.println("\nFeatures: "+features+"\nIteration "+i);

            System.out.println("Train");
            DoubleMatrix res = compute(input, batchSize, labels);
            if(cost > previousCost) alpha *= 0.75;
            boolean finished = compareResults(Utils.computeResults(res), labels);
            System.out.println("Test");
            DoubleMatrix testRes = compute(test, testLab);
            compareResults(Utils.computeResults(testRes), testLab);
            for(int j = 0; j < cls.length; j++) {
                System.out.println("ConvA"+j+": " + cls[j].getA());
            }
            for(int j = 0; j < lds.length; j++) {
                System.out.println("FC"+j+": " + lds[j].getA());
            }
            //if(finished) break;
        }
        write(name);
    }


    public boolean compareResults(int[][] result, DoubleMatrix labels) {
        double[] sums = new double[result[0].length];
        System.out.println(result.length+":"+labels.rows+":"+labels.columns);
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                sums[j] += labels.get(i, result[i][j]);
            }
        }
        for(int i = 0; i < sums.length; i++) {
            if(sums[i] > 0) System.out.println(i+": "+sums[i]+"/"+result.length+ " = " + (sums[i]/result.length));
        }
        return(sums[0]==result.length);
    }

    public double computeCost(DoubleMatrix[][] input, DoubleMatrix labels) {
        DoubleMatrix[][][] convResults = new DoubleMatrix[cls.length+1][][];
        convResults[0] = input;
        for(int k = 0; k < cls.length; k++) {
            convResults[k+1] = cls[k].compute(convResults[k]);
        }
        DoubleMatrix[] ldsResults = new DoubleMatrix[lds.length+1];
        ldsResults[0] = Utils.flatten(convResults[cls.length]);
        for(int k = 0; k < lds.length; k++) {
            ldsResults[k+1] = lds[k].compute(ldsResults[k]);
        }
        DoubleMatrix p = MatrixFunctions.log(ldsResults[ldsResults.length-1]);
        return -p.mul(labels).sum() / input.length;
    }

    public void gradientCheck(DoubleMatrix[][] input, DoubleMatrix labels) {
        DoubleMatrix[][][] convResults = new DoubleMatrix[cls.length+1][][];
        convResults[0] = input;
        for(int i = 0; i < cls.length; i++) {
            convResults[i+1] = cls[i].feedForward(convResults[i]);
        }
        DoubleMatrix[] ldsResults = new DoubleMatrix[lds.length+1];
        ldsResults[0] = Utils.flatten(convResults[cls.length]);
        for(int i = 0; i < lds.length; i++) {
            ldsResults[i+1] = lds[i].feedforward(ldsResults[i]);
        }
        Gradients cr;
        DoubleMatrix delta = ldsResults[ldsResults.length-1].sub(labels);
        for(int k = lds.length-1; k >= 0; k--) {
            cr = lds[k].cost(ldsResults[k], ldsResults[k+1], delta, labels);
            delta = cr.delta;
            lds[k].gradientCheck(input, labels, cr, this);
        }
        DoubleMatrix[][] delt = Utils.expand(delta, convResults[cls.length][0].length, convResults[cls.length][0][0].rows, convResults[cls.length][0][0].columns);
        for(int k = cls.length-1; k >= 0; k--) {
            cr = cls[k].cost(convResults[k], convResults[k+1], delt);

            delt = cls[k].gradientCheck(cr, input, labels, this);
        }
    }

    private void shuffleIndices(int[] indices, int iterations) {
        for(int i = 0; i < iterations; i++) {
            int a = r.nextInt(indices.length);
            int b = r.nextInt(indices.length);
            int temp = indices[a];
            indices[a] = indices[b];
            indices[b] = temp;
        }
    }


    public DoubleMatrix compute(DoubleMatrix[][] input, int batchSize) {
        DoubleMatrix res = null;
        for(int j = 0; j < input.length/batchSize; j++) {
            System.out.print(j + " ");
            DoubleMatrix[][] in = new DoubleMatrix[batchSize][];
            for(int k = 0; k < batchSize; k++) {
                in[k] = input[j*batchSize+k];
            }
            for (int i = 0; i < cls.length; i++) {
                in = cls[i].compute(in);
            }
            DoubleMatrix fin = Utils.flatten(in);
            for (int i = 0; i < lds.length; i++) {
                fin = lds[i].compute(fin);
            }
            if (res == null) res = fin;
            else res = DoubleMatrix.concatVertically(res, fin);
        }
        return res;
    }

    public DoubleMatrix compute(DoubleMatrix[][] input, int batchSize, DoubleMatrix labels) {
        DoubleMatrix res = null;
        cost = 0;
        for(int j = 0; j < input.length/batchSize; j++) {
            DoubleMatrix[][] in = new DoubleMatrix[batchSize][];
            DoubleMatrix labs = labels.getRange(j*batchSize, j*batchSize+batchSize, 0, labels.columns);
            for(int k = 0; k < batchSize; k++) {
                in[k] = input[j*batchSize+k];
            }
            for (int i = 0; i < cls.length; i++) {
                in = cls[i].compute(in);
            }
            DoubleMatrix fin = Utils.flatten(in);
            for (int i = 0; i < lds.length-1; i++) {
                fin = lds[i].compute(fin);
            }
            DoubleMatrix out = lds[lds.length-1].compute(fin);
            if (res == null) res = out;
            else res = DoubleMatrix.concatVertically(res, out);
            DoubleMatrix p = MatrixFunctions.log(out);
            cost += -p.mul(labs).sum() / (out.rows*batchSize);
        }
        System.out.println("Cost: "+cost);
        return res;
    }

    public DoubleMatrix compute(DoubleMatrix[][] input, DoubleMatrix labels) {
        for (int i = 0; i < cls.length; i++) {
            input = cls[i].compute(input);
        }
        DoubleMatrix fin = Utils.flatten(input);
        for (int i = 0; i < lds.length-1; i++) {
            fin = lds[i].compute(fin);
        }
        DoubleMatrix out = lds[lds.length-1].compute(fin);
        DoubleMatrix p = MatrixFunctions.log(out);
        cost = -p.mul(labels).sum() / out.rows;
        System.out.println("Cost: "+cost);
        return out;
    }

    public void write(String filename) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
            DeviceStructuredLayer[] cp = new DeviceStructuredLayer[cls.length];
            DeviceFullyConnectedLayer[] fc = new DeviceFullyConnectedLayer[lds.length];
            for(int i = 0; i < cls.length; i++) {
                cp[i] = cls[i].getDevice();
            }
            for(int i = 0; i < lds.length; i++) {
                fc[i] = lds[i].getDevice();
            }
            DeviceNeuralNetwork nn = new DeviceNeuralNetwork(cp, fc);
            out.writeObject(nn);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
