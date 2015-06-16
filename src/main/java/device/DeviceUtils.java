package device;

import Jama.Matrix;
import Jama.SingularValueDecomposition;
import org.jtransforms.fft.DoubleFFT_2D;

/**
 * DeviceUtils - Utilities for the device neural network package
 *
 * @author Timothy Jassmann
 * @version 06/02/2015
 */
public abstract class DeviceUtils {
    public static final int NONE = 0;
    public static final int SIGMOID = 1;
    public static final int PRELU = 2;
    public static final int RELU = 3;
    public static final int SOFTMAX = 4;

    /**
     * activationFunction - performs an activation function on input matrix
     *
     * @param type Type of activation function
     * @param input Input matrix
     * @param a A value of layer
     *
     * @return Result of activation function on input
     */
    public static Matrix activationFunction(int type, Matrix input, double a) {
        switch(type) {
            case SIGMOID:
                return sigmoid(input);
            case PRELU:
                return prelu(input, a);
            case RELU:
                return relu(input);
            case SOFTMAX:
                return softmax(input);
            case NONE:
            default:
                return input;
        }
    }

    /**
     * computeRanking - ranks results in order of percent likelihood
     *
     * @param result matrix of percent likelihood of results
     *
     * @return int array containing ranking of classification of each result
     */
    public static int[] computeRanking(Matrix result) {
        int[] results = new int[result.getColumnDimension()];
        double[] values = new double[result.getColumnDimension()];
        for(int element = 0; element < result.getColumnDimension(); element++) {
            for(int rank = 0; rank < result.getColumnDimension(); rank++) {
                if(result.get(0, element) > values[rank]) {
                    //move everything below rank down one
                    for(int l = result.getColumnDimension() - 1; l > rank; l--) {
                        values[l] = values[l - 1];
                        results[l] = results[l - 1];
                    }
                    values[rank] = result.get(0, element);
                    results[rank] = element;
                    break;
                }
            }
        }
        return results;
    }

    /**
     * convolve - convolves two matrices
     *
     * @param input input to be convolved over
     * @param kernel kernel to convolve over input
     *
     * @return Result of convolution
     */
    public static Matrix convolve(Matrix input, Matrix kernel) {
        //Flip Kernel
        Matrix flippedKernel = new Matrix(kernel.getRowDimension(), kernel.getColumnDimension());
        for(int i = 0; i < kernel.getRowDimension(); i++) {
            for(int j = 0; j < kernel.getColumnDimension(); j++) {
                flippedKernel.set(i, j, kernel.get(kernel.getRowDimension() - 1 - i,
                        kernel.getColumnDimension() - 1 - j));
            }
        }
        //Constants
        int totalRows = input.getRowDimension() + flippedKernel.getRowDimension() - 1;
        int totalCols = input.getColumnDimension() + flippedKernel.getColumnDimension() - 1;
        int rowSize = input.getRowDimension() - flippedKernel.getRowDimension() + 1;
        int colSize = input.getColumnDimension() - flippedKernel.getColumnDimension() + 1;
        //Transition input and kernel to larger matrices
        double[][] in = new double[totalRows][totalCols * 2];
        double[][] kern = new double[totalRows][totalCols * 2];
        for(int i = 0; i < input.getRowDimension(); i++) {
            for(int j = 0; j < input.getColumnDimension(); j++) {
                in[i][j] = input.get(i, j);
            }
        }
        for(int i = 0; i < flippedKernel.getRowDimension(); i++) {
            for(int j = 0; j < flippedKernel.getColumnDimension(); j++) {
                kern[i][j] = flippedKernel.get(i, j);
            }
        }
        //DFT on input and kernel
        DoubleFFT_2D t = new DoubleFFT_2D(totalRows, totalCols);
        t.realForwardFull(in);
        t.realForwardFull(kern);
        //complex multiplication of input and kernel
        double[][] res = complexMult(in, kern);
        //Inverse of DFT on result
        t.complexInverse(res, true);
        //Transition res back into result matrix
        Matrix result = new Matrix(rowSize, colSize);
        for(int i = 0; i < rowSize; i++) {
            for(int j = 0; j < colSize; j++) {
                result.set(i, j,
                        res[(totalRows - rowSize) / 2 + i][((totalCols - colSize) / 2 + j) * 2]);
            }
        }
        return result;
    }

    /**
     * flatten - flattens an array of matrices into a row vector
     *
     * @param input Matrix to flatten
     *
     * @return Flattened matrix
     */
    public static Matrix flatten(Matrix[] input) {
        Matrix image = new Matrix(1, input.length * input[0].getRowDimension()
                * input[0].getColumnDimension());
        for(int channel = 0; channel < input.length; channel++) {
            for(int row = 0; row < input[channel].getRowDimension(); row++) {
                for(int column = 0; column < input[channel].getColumnDimension(); column++) {
                    image.set(0, channel * input[channel].getRowDimension() * input[channel].getColumnDimension()
                            +  row * input[channel].getColumnDimension() + column, input[channel].get(row,column));
                }
            }
        }
        return image;
    }

    /**
     * normalizeData - normalizes data to have zero mean and unit variance
     *
     * @param data Input matrices representing each channel of the input
     *
     * @return normalized data
     */
    public static Matrix[] normalizeData(Matrix[] data) {
        for (Matrix channel : data) {
            channel.minusEquals(new Matrix(channel.getRowDimension(), channel.getColumnDimension(), mean(channel)));
            double var = mean(channel.arrayTimes(channel));
            double stdev = Math.sqrt(var);
            channel.timesEquals(1 / stdev);
        }
        return data;
    }

    /**
     * ZCAWhiten - performs ZCA whitening on input matrix
     *
     * @param input Matrix to be whitened
     * @param epsilon Epsilon value for ZCA whitening
     *
     * @return ZCA whitened input
     */
    public static Matrix ZCAWhiten(Matrix input, double epsilon) {
        double mean = mean(input);
        input.minusEquals(new Matrix(input.getRowDimension(), input.getColumnDimension(), mean));
        Matrix sigma = input.times(input.transpose()).times(1.0 / input.getColumnDimension());
        SingularValueDecomposition svd = sigma.svd();
        Matrix s = svd.getS();
        for(int i = 0; i < s.getRowDimension(); i++) {
            s.set(i, i, 1 / (Math.sqrt(s.get(i, i) + epsilon)));
        }
        return svd.getU().times(s).times(svd.getU().transpose()).times(input);
    }

    /**
     * complexMult - performs an elementwise multiplication of a 2d array of complex numbers
     *              where each even column is the real part and each odd column is
     *              the imaginary part.
     *
     * @param a A 2d array of complex numbers to be multiplied
     * @param b A 2d array of complex numbers to be multiplied
     *
     * @return Result of elementwise multiplication of a and b
     */
    private static double[][] complexMult(double[][] a, double[][] b) {
        double[][] res = new double[a.length][a[0].length];
        for(int row = 0; row < a.length; row++) {
            for(int column = 0; column < a[row].length; column+=2) {
                res[row][column] = a[row][column] * b[row][column] - (a[row][column + 1] * b[row][column + 1]);
                res[row][column + 1] = a[row][column] * b[row][column + 1] + (a[row][column + 1] * b[row][column]);
            }
        }
        return res;
    }

    /**
     * max - calculates the max of a matrix
     *
     * @param input matrix to calculate max of
     *
     * @return max of input
     */
    private static double max(Matrix input) {
        double max = input.get(0, 0);
        for(int i = 0; i < input.getRowDimension(); i++) {
            for(int j = 0; j < input.getColumnDimension(); j++) {
                if(max < input.get(i, j)) max = input.get(i, j);
            }
        }
        return max;
    }

    /**
     * mean - calculates the mean of a matrix
     *
     * @param input matrix to calculate the mean of
     *
     * @return mean of input
     */
    private static double mean(Matrix input) {
        double mean = 0;
        for(int i = 0; i < input.getRowDimension(); i++) {
            for(int j = 0; j < input.getColumnDimension(); j++) {
                mean += input.get(i, j);
            }
        }
        return mean / (input.getRowDimension() * input.getColumnDimension());
    }

    /**
     * prelu - The PReLU activation function
     *
     * @param input The input to the PReLU function
     * @param a The a value for the PReLU function
     *
     * @return The result of applying the PReLU function to input with a
     */
    private static Matrix prelu(Matrix input, double a) {
        for(int row = 0; row < input.getRowDimension(); row++) {
            for(int column = 0; column < input.getColumnDimension(); column++) {
                if(input.get(row, column) < 0) input.set(row, column, a * input.get(row, column));
            }
        }
        return input;
    }

    /**
     * prelu - The ReLU activation function
     *
     * @param input The input to the ReLU function
     *
     * @return The result of applying the ReLU function to input with a
     */
    private static Matrix relu(Matrix input) {
        for(int row = 0; row < input.getRowDimension(); row++) {
            for(int column = 0; column < input.getColumnDimension(); column++) {
                if(input.get(row, column) < 0) input.set(row, column, 0);
            }
        }
        return input;
    }

    /**
     * sigmoid - The sigmoid activation function
     *
     * @param input The input to the sigmoid function
     *
     * @return The result of applying the sigmoid function to input
     */
    private static Matrix sigmoid(Matrix input) {
        for(int row = 0; row < input.getRowDimension(); row++) {
            for(int column = 0; column < input.getColumnDimension(); column++) {
                input.set(row, column, 1 / (1 + Math.exp(-input.get(row, column))));
            }
        }
        return input;
    }

    /**
     * softmax - the softmax activation function
     *
     * @param input The input to the softmax function
     *
     * @return The result of applying the softmax function to input
     */
    private static Matrix softmax(Matrix input) {
        double max = max(input);
        for(int row = 0; row < input.getRowDimension(); row++) {
            for (int column = 0; column < input.getColumnDimension(); column++) {
                input.set(row, column, Math.exp(input.get(row, column) - max));
            }
            double sum = 0;
            for (int column = 0; column < input.getColumnDimension(); column++) {
                sum += input.get(row, column);
            }
            for (int column = 0; column < input.getColumnDimension(); column++) {
                input.set(row, column, input.get(row, column) / sum);
            }
        }

        return input;
    }
}

