package nn;

import org.jblas.ComplexDoubleMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.Singular;
import org.jtransforms.fft.DoubleFFT_2D;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@SuppressWarnings("StatementWithEmptyBody")
/**
 * Utils - Utilities for the neural network package
 *
 * @author Timothy Jassmann
 * @version 06/16/2015
 */
public abstract class Utils {
    public static final int NUMTHREADS = 8;
    public static final int NONE = 0;
    public static final int SIGMOID = 1;
    public static final int PRELU = 2;
    public static final int RELU = 3;
    public static final int SOFTMAX = 4;


    /**
     * activationFunction - computes an activation function on input
     *
     * @param type type of activation function
     * @param input matrix to perform activation function on
     * @param a a value of layer to perform activation on
     *
     * @return result of applying activation on input
     */
    public static DoubleMatrix activationFunction(int type, DoubleMatrix input, double a) {
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
     * activationGradient - computes the activation function gradient on input
     *
     * @param type type of activation function
     * @param input matrix to perform activation function on
     * @param a a value of layer to perform activation on
     *
     * @return gradient of activation function on input
     */
    public static DoubleMatrix activationGradient(int type, DoubleMatrix input, double a) {
        switch(type) {
            case SIGMOID:
                return sigmoidGradient(input);
            case PRELU:
                return preluGradient(input, a);
            case RELU:
                return reluGradient(input);
            case NONE:
            default:
                return DoubleMatrix.ones(input.rows, input.columns);
        }
    }

    /**
     * aGradient - calculates the gradient of the a value
     *
     * @param type type of activation function
     * @param result result of layer
     * @param delta gradient propagated back to layer
     *
     * @return gradient of a value
     */
    public static double aGradient(int type, DoubleMatrix[][] result, DoubleMatrix[][] delta) {
        double aGrad = 0;
        switch(type) {
            case PRELU:
                for(int i = 0; i < result.length; i++) {
                    for(int j = 0; j < result[i].length; j++) {
                        for(int k = 0; k < result[i][j].length; k++) {
                            if(result[i][j].get(k) <= 0) aGrad += result[i][j].get(k) * delta[i][j].get(k);
                        }
                    }
                }
                return aGrad/result.length;
            case SIGMOID:
            case RELU:
            case NONE:
            default: return aGrad;
        }
    }

    /**
     * aGradient - calculates the gradient of the a value
     *
     * @param type type of activation function
     * @param result result of layer
     * @param delta gradient propagated back to layer
     *
     * @return gradient of a value
     */
    public static double aGradient(int type, DoubleMatrix result, DoubleMatrix delta) {
        double aGrad = 0;
        switch(type) {
            case PRELU:
                for(int i = 0; i < result.length; i++) {
                    if( result.get(i) <= 0) aGrad += result.get(i)*delta.get(i);
                }
                return aGrad/result.rows;
            case SIGMOID: return 0;
            case RELU: return 0;
            case NONE: return 0;
            default: return 0;
        }
    }

    /**
     * alterImages - performs pixel alterations and random rotations to images
     *
     * @param images images to alter
     */
    public static void alterImages(DoubleMatrix[][] images) {
        class AlterThread implements Runnable {
            private DoubleMatrix[] image;

            public AlterThread(DoubleMatrix[] image) {
                this.image = image;
            }

            @Override
            public void run() {
                DoubleMatrix a = DoubleMatrix.randn(image.length).mul(0.1);
                DoubleMatrix i = new DoubleMatrix(image.length, image[0].length);
                for(int k = 0; k < image.length; k++) {
                    for (int j = 0; j < image[0].length; j++) {
                        i.put(k, j, image[k].get(j));
                    }
                }
                i.subiColumnVector(i.rowMeans());
                i = i.mmul(i.transpose()).div(i.columns);
                DoubleMatrix[] svd = Singular.fullSVD(i);
                svd[1].muli(a);
                DoubleMatrix res = svd[0].mmul(svd[1]);
                for(int j = 0; j < image.length; j++) {
                    image[j].addi(res.get(j));
                }
                if(Math.random() > 0.5) {
                    for (int j = 0; j < image.length; j++) {
                        image[j] = Utils.flipHorizontal(image[j]);
                    }
                }
                if(Math.random() > 0.5) {
                    for (int j = 0; j < image.length; j++) {
                        image[j] = Utils.flipVertical(image[j]);
                    }
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(DoubleMatrix[] image : images) {
            Runnable worker = new AlterThread(image);
            executor.execute(worker);
        }
        executor.shutdown();
        while(!executor.isTerminated());
    }

    /**
     * computeRanking - ranks results in order of percent likelihood
     *
     * @param result matrix of percent likelihood of results
     *
     * @return 2d int array containing ranking of classification of each result for each image
     */
    public static int[][] computeRanking(DoubleMatrix result) {
        int[][] results = new int[result.rows][result.columns];
        for(int i = 0; i < result.rows; i++) {
            double[] current = new double[result.columns];
            for(int j = 0; j < result.columns; j++) {
                for(int k = 0; k < result.columns; k++) {
                    if(result.get(i,j) > current[k]) {
                        for(int l = result.columns - 1; l > k; l--) {
                            current[l] = current[l - 1];
                            results[i][l] = results[i][l - 1];
                        }
                        current[k] = result.get(i,j);
                        results[i][k] = j;
                        break;
                    }
                }
            }
        }
        return results;
    }

    /**
     * convolve - convolves kernel over input
     *
     * @param input matrix to convolve over
     * @param kernel kernel to convolve
     * @param valid performs a valid convolution if true, otherwise performs full convolution
     *
     * @return result of convolving kernel over input
     */
    public static DoubleMatrix convolve(DoubleMatrix input, DoubleMatrix kernel, boolean valid) {
        //Constants
        int totalRows = input.rows + kernel.rows - 1;
        int totalCols = input.columns + kernel.columns - 1;
        int rowSize = input.rows - kernel.rows + 1;
        int colSize = input.columns - kernel.columns + 1;
        //reverse kernel
        kernel = reverseMatrix(kernel);
        //Transition input and kernel to larger matrices
        input = DoubleMatrix.concatHorizontally(input, DoubleMatrix.zeros(input.rows, kernel.columns - 1));
        input = DoubleMatrix.concatVertically(input, DoubleMatrix.zeros(kernel.rows - 1, input.columns));
        kernel = DoubleMatrix.concatHorizontally(kernel, DoubleMatrix.zeros(kernel.rows, input.columns - kernel.columns));
        kernel = DoubleMatrix.concatVertically(kernel, DoubleMatrix.zeros(input.rows - kernel.rows,kernel.columns));
        //Transition input and kernel to complex matrices
        ComplexDoubleMatrix inputDFT = new ComplexDoubleMatrix(input);
        ComplexDoubleMatrix kernelDFT = new ComplexDoubleMatrix(kernel);
        DoubleFFT_2D t = new DoubleFFT_2D(inputDFT.columns, inputDFT.rows);
        //DFT on input and kernel
        t.complexForward(inputDFT.data);
        t.complexForward(kernelDFT.data);
        //complex multiplication of input and kernel
        kernelDFT.muli(inputDFT);
        //Inverse of DFT on result
        t.complexInverse(kernelDFT.data, true);
        DoubleMatrix result = kernelDFT.getReal();
        if(!valid) return result;
        else {
            //Resize result
            int startRows = (totalRows - rowSize) / 2;
            int startCols = (totalCols - colSize) / 2;
            result = result.getRange(startRows, startRows + rowSize, startCols, startCols + colSize);
            return result;
        }
    }

    /**
     * expand - expands a flattened matrix back into the shape of images
     *
     * @param input the matrix to expand
     * @param channels the number of channels to expand to
     * @param rows the number of rows to expand to
     * @param cols the number of columns to expand to
     *
     * @return expanded matrix
     */
    protected static DoubleMatrix[][] expand(DoubleMatrix input, int channels, int rows, int cols) {
        DoubleMatrix[][] result = new DoubleMatrix[input.rows][channels];
        for(int i = 0; i < input.rows; i++) {
            for(int j = 0; j < channels; j++) {
                result[i][j] = new DoubleMatrix(rows, cols);
                for(int k = 0; k < rows; k++) {
                    for(int l = 0; l < cols; l++) {
                        result[i][j].put(k, l, input.get(i, j * rows * cols + k * cols + l));
                    }
                }
            }
        }
        return result;
    }

    /**
     * flatten - flattens a 2d array of structured images to a single matrix
     *
     * @param input the matrices to flatten
     *
     * @return matrix containing all data in input
     */
    public static DoubleMatrix flatten(DoubleMatrix[][] input) {
        DoubleMatrix images = null;
        for (DoubleMatrix[] image : input) {
            DoubleMatrix row = null;
            for (DoubleMatrix channel : image) {
                for (int k = 0; k < channel.rows; k++) {
                    if (row == null) row = channel.getRow(k);
                    else row = DoubleMatrix.concatHorizontally(row, channel.getRow(k));
                }
            }
            if(row != null) {
                if (images == null) images = row;
                else images = DoubleMatrix.concatVertically(images, row);
            }
        }
        return images;
    }

    /**
     * normalizeData - normalizes the input data to have zero mean and unit variance
     *                 for each channel for each image
     *
     * @param data data to normalize
     *
     * @return normalized data
     */
    public static DoubleMatrix[][] normalizeData(DoubleMatrix[][] data) {
        for (DoubleMatrix[] image : data) {
            for (DoubleMatrix channel : image) {
                channel.subi(channel.mean());
                double var = channel.mul(channel).mean();
                double stdev = Math.sqrt(var);
                channel.divi(stdev);
            }
        }
        return data;
    }

    /**
     * reverseMatrix - reverses the rows and columns of a matrix
     *
     * @param input matrix to reverse
     *
     * @return reversed matrix
     */
	public static DoubleMatrix reverseMatrix(DoubleMatrix input) {
		return flipVertical(flipHorizontal(input));
	}

    /**
     * samplePatches - randomly samples patches from images
     *
     * @param patchRows rows of patches
     * @param patchCols columns of patches
     * @param numPatches number of patches
     * @param images images to sample from
     *
     * @return Matrix of sampled patches
     */
    public static DoubleMatrix samplePatches(final int patchRows, final int patchCols, final int numPatches, final DoubleMatrix[][] images) {
        final DoubleMatrix patches = new DoubleMatrix(numPatches, images[0].length*patchRows*patchCols);
        class Patcher implements Runnable {
            private int threadNo;

            public Patcher(int threadNo) {
                this.threadNo = threadNo;
            }

            @Override
            public void run() {
                int count = numPatches / Utils.NUMTHREADS;
                for (int i = 0; i < count; i++) {
                    Random rand = new Random();
                    int randomImage = rand.nextInt(images.length);
                    int randomY = rand.nextInt(images[randomImage][0].rows - patchRows + 1);
                    int randomX = rand.nextInt(images[randomImage][0].columns - patchCols + 1);
                    DoubleMatrix ch = null;
                    for (int j = 0; j < images[randomImage].length; j++) {
                        DoubleMatrix patch = images[randomImage][j].getRange(randomY, randomY + patchRows, randomX, randomX + patchCols);
                        patch = patch.reshape(1, patchRows * patchCols);
                        if(ch == null) ch = patch;
                        else ch = DoubleMatrix.concatHorizontally(ch, patch);
                    }
                    if(ch != null) patches.putRow(threadNo*count + i, ch);
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int i = 0; i < Utils.NUMTHREADS; i++) {
            Runnable patcher = new Patcher(i);
            executor.execute(patcher);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return patches;
    }

    /**
     * visualizeColorImg - creates a png out of a color image split by channels in doublematrix arrays
     *
     * @param img image to create png out of
     * @param filename filename of image
     * @throws IOException
     */
    public static void visualizeColorImg(DoubleMatrix[] img, String filename) throws IOException {
        int width = img[0].columns;
        int height = img[0].rows;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        double min = img[0].min();
        double max = img[0].max();
        for (DoubleMatrix channel : img) {
            min = channel.min() < min ? channel.min() : min;
            max = channel.max() > max ? channel.max() : max;
        }
        max = max - min;
        for (DoubleMatrix channel : img) {
            channel.subi(min);
            channel.divi(max);
            channel.muli(255);
        }
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                int col = ((int)img[2].get(i,j) << 16) | ((int)img[1].get(i,j) << 8) | (int)img[0].get(i,j);
                image.setRGB(j,i, col);
            }
        }
        File imageFile = new File(filename+".png");
        ImageIO.write(image, "png", imageFile);
    }

    /**
     * ZCAWhiten - performs ZCA Whitening on input image
     *
     * @param input images to perform ZCA Whitening on
     * @param epsilon epsilon value for ZCA Whitening
     *
     * @return ZCA Whitened image
     */
    public static DoubleMatrix[][] ZCAWhiten(DoubleMatrix[][] input, double epsilon) {
        DoubleMatrix img = flatten(input);
        img.subiColumnVector(img.rowMeans());
        DoubleMatrix sigma = img.mmul(img.transpose()).div(img.columns);
        DoubleMatrix[] svd = Singular.fullSVD(sigma);
        DoubleMatrix s = DoubleMatrix.diag(MatrixFunctions.sqrt(svd[1].add(epsilon)).rdiv(1));
        DoubleMatrix res = svd[0].mmul(s).mmul(svd[0].transpose()).mmul(img);
        return expand(res, input[0].length, input[0][0].rows, input[0][0].columns);
    }


    /**
     * flipHorizontal - flip the values of a matrix horizontally
     *
     * @param input the matrix to flip
     *
     * @return the flipped matrix
     */
    private static DoubleMatrix flipHorizontal(DoubleMatrix input) {
        DoubleMatrix result = input.dup();
        for(int i = 0; i < result.rows / 2; i++) {
            result.swapRows(i, result.rows - i - 1);
        }
        return result;
    }

    /**
     * flipVertical - flip the values of a matrix vertically
     *
     * @param input the matrix to flip
     *
     * @return the flipped matrix
     */
    private static DoubleMatrix flipVertical(DoubleMatrix input) {
        DoubleMatrix result = input.dup();
        for(int i = 0; i < result.columns / 2; i++) {
            result.swapColumns(i, result.columns - i - 1);
        }
        return result;
    }

    /**
     * prelu - the PReLU activation function
     *
     * @param input the input to the function
     * @param a the a value of the layer
     *
     * @return the result of applying PReLU to input
     */
    private static DoubleMatrix prelu(DoubleMatrix input, double a) {
        DoubleMatrix result = new DoubleMatrix(input.rows, input.columns);
        for(int i = 0; i < input.rows; i++) {
            for(int j = 0; j < input.columns; j++) {
                if(input.get(i, j) > 0) result.put(i, j, input.get(i, j));
                else result.put(i, j, a * input.get(i, j));
            }
        }
        return result;
    }

    /**
     * preluGradeint - calculates the gradient of the PReLU activation function
     *
     * @param input the input to the function
     * @param a the a value of the layer
     *
     * @return the gradient of the PReLU activation function applied to input
     */
    private static DoubleMatrix preluGradient(DoubleMatrix input, double a) {
        DoubleMatrix result = new DoubleMatrix(input.rows, input.columns);
        for(int i = 0; i < input.rows; i++) {
            for(int j = 0; j < input.columns; j++) {
                if(input.get(i, j) < 0) result.put(i, j, a);
                else result.put(i, j, 1);
            }
        }
        return result;
    }

    /**
     * relu - the ReLU activation function
     *
     * @param input the input to the function
     *
     * @return the result of applying ReLU to input
     */
    private static DoubleMatrix relu(DoubleMatrix input) {
        DoubleMatrix result = new DoubleMatrix(input.rows, input.columns);
        for(int i = 0; i < input.rows; i++) {
            for(int j = 0; j < input.columns; j++) {
                if(input.get(i, j) > 0) result.put(i, j, input.get(i, j));
            }
        }
        return result;
    }

    /**
     * reluGradeint - calculates the gradient of the ReLU activation function
     *
     * @param input the input to the function
     *
     * @return the gradient of the ReLU activation function applied to input
     */
    private static DoubleMatrix reluGradient(DoubleMatrix input) {
        return input.gt(0);
    }

    /**
     * sigmoid - the sigmoid activation function
     *
     * @param input the input to the function
     *
     * @return the result of applying sigmoid to input
     */
    private static DoubleMatrix sigmoid(DoubleMatrix input) {
        return MatrixFunctions.exp(input.neg()).add(1).rdiv(1);
    }

    /**
     * sigmoidGradient - calculates the gradient of the sigmoid activation function
     *
     * @param input the input to the function
     *
     * @return the gradient of the sigmoid activation function applied to input
     */
    private static DoubleMatrix sigmoidGradient(DoubleMatrix input) {
        return input.rsub(1).mul(input);
    }

    /**
     * softmax - the softmax activation function
     *
     * @param input the input to the function
     *
     * @return the result of applying softmax to input
     */
    private static DoubleMatrix softmax(DoubleMatrix input) {
        DoubleMatrix result = input.subColumnVector(input.rowMaxs());
        MatrixFunctions.expi(result);
        result.diviColumnVector(result.rowSums());
        return result;
    }
}
