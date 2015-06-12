package device;

import Jama.Matrix;
import Jama.SingularValueDecomposition;
import org.jtransforms.fft.DoubleFFT_2D;

/**
 * Created by jassmanntj on 4/13/2015.
 */
public abstract class DeviceUtils {
    public static final int NONE = 0;
    public static final int SIGMOID = 1;
    public static final int PRELU = 2;
    public static final int RELU = 3;
    public static final int SOFTMAX = 4;

    public static Matrix[] normalizeData(Matrix[] data) {
        for(int j = 0; j < data.length; j++) {
            data[j].minusEquals(new Matrix(data[j].getRowDimension(), data[j].getColumnDimension(), mean(data[j])));
            double var = mean(data[j].arrayTimes(data[j]));
            double stdev = Math.sqrt(var);
            data[j].timesEquals(1 / stdev);
        }
        return data;
    }

    private static double mean(Matrix m) {
        double mean = 0;
        for(int i = 0; i < m.getRowDimension(); i++) {
            for(int j = 0; j < m.getColumnDimension(); j++) {
                mean += m.get(i,j);
            }
        }
        return mean/(m.getRowDimension()*m.getColumnDimension());
    }

    public static Matrix conv2d(Matrix input, Matrix kernel) {
        Matrix flippedKernel = new Matrix(kernel.getRowDimension(), kernel.getColumnDimension());
        for(int i = 0; i < kernel.getRowDimension(); i++) {
            for(int j = 0; j < kernel.getColumnDimension(); j++) {
                flippedKernel.set(i,j, kernel.get(kernel.getRowDimension()-1-i,kernel.getColumnDimension()-1-j));
            }
        }
        kernel = flippedKernel;
        int totalRows = input.getRowDimension() + kernel.getRowDimension() - 1;
        int totalCols = input.getColumnDimension() + kernel.getColumnDimension() - 1;
        int rowSize = input.getRowDimension() - kernel.getRowDimension() + 1;
        int colSize = input.getColumnDimension() - kernel.getColumnDimension() + 1;
        int startRows = (totalRows-rowSize)/2;
        int startCols = (totalCols-colSize)/2;
        double[][] in = new double[totalRows][totalCols*2];
        double[][] kern = new double[totalRows][totalCols*2];
        for(int i = 0; i < input.getRowDimension(); i++) {
            for(int j = 0; j < input.getColumnDimension(); j++) {
                in[i][j] = input.get(i,j);
            }
        }
        for(int i = 0; i < kernel.getRowDimension(); i++) {
            for(int j = 0; j < kernel.getColumnDimension(); j++) {
                kern[i][j] = kernel.get(i,j);
            }
        }

        DoubleFFT_2D t = new DoubleFFT_2D(totalRows, totalCols);
        t.realForwardFull(in);
        t.realForwardFull(kern);
        double[][] res = complexMult(in, kern);

        t.complexInverse(res, true);

        Matrix result = new Matrix(rowSize, colSize);
        for(int i = 0; i < rowSize; i++) {
            for(int j = 0; j < colSize; j++) {
                result.set(i,j,res[startRows+i][(startCols+j)*2]);
            }
        }
        return result;
    }

    public static Matrix ZCAWhiten(Matrix input, double epsilon) {
        double mean = 0;
        for(int j = 0; j < input.getColumnDimension(); j++) {
            mean += input.get(0,j);
        }
        mean /= input.getRowDimension()*input.getColumnDimension();
        input.minusEquals(new Matrix(input.getRowDimension(), input.getColumnDimension(), mean));
        Matrix sigma = input.times(input.transpose()).times(1.0/input.getColumnDimension());
        sigma.arrayTimesEquals(Matrix.identity(sigma.getRowDimension(),sigma.getColumnDimension()));
        SingularValueDecomposition svd = sigma.svd();
        Matrix s = svd.getS();
        for(int i = 0; i < s.getRowDimension(); i++) {
            s.set(i, i, 1/(Math.sqrt(s.get(i, i)+epsilon)));
        }
        Matrix res = svd.getU().times(s).times(svd.getU().transpose()).times(input);

        return res;
    }

    private static double[][] complexMult(double[][] a, double[][] b) {
        double[][] res = new double[a.length][a[0].length];
        for(int i = 0; i < a.length; i++) {
            for(int j = 0; j < a[i].length; j+=2) {
                res[i][j] = a[i][j] * b[i][j] - (a[i][j+1] * b[i][j+1]);
                res[i][j+1] = a[i][j] * b[i][j+1] + (a[i][j+1] * b[i][j]);
            }
        }
        return res;
    }

    public static int[] computeResults(Matrix result) {
        int[] results = new int[result.getColumnDimension()];
        double[] current = new double[result.getColumnDimension()];
        for(int j = 0; j < result.getColumnDimension(); j++) {
            for(int k = 0; k < result.getColumnDimension(); k++) {
                if(result.get(0,j) > current[k]) {
                    for(int l = result.getColumnDimension()-1; l > k; l--) {
                        current[l] = current[l-1];
                        results[l] = results[l-1];
                    }
                    current[k] = result.get(0,j);
                    results[k] = j;
                    break;
                }
            }
        }
        return results;
    }


    public static Matrix flatten(Matrix[] z) {
        Matrix image = new Matrix(1, z.length*z[0].getRowDimension()*z[0].getColumnDimension());
        for(int i = 0; i < z.length; i++) {
            for(int j = 0; j < z[i].getRowDimension(); j++) {
                for(int k = 0; k < z[i].getColumnDimension(); k++) {
                    image.set(0, i*z[i].getRowDimension()*z[i].getColumnDimension()+j*z[i].getColumnDimension()+k,z[i].get(j,k));
                }
            }
        }
        return image;
    }

    public static Matrix activationFunction(int type, Matrix z, double a) {
        switch(type) {
            case SIGMOID:
                return sigmoid(z);
            case PRELU:
                return prelu(z, a);
            case RELU:
                return relu(z);
            case SOFTMAX:
                return softmax(z);
            case NONE:
                return z;
            default:
                return sigmoid(z);
        }
    }

    private static Matrix softmax(Matrix z) {
        double max = z.norm1();
        Matrix res = new Matrix(z.getRowDimension(), z.getColumnDimension());
        for(int j = 0; j < z.getColumnDimension(); j++) {
            res.set(0,j,Math.exp(z.get(0,j)-max));
        }
        double sum = 0;
        for(int i = 0; i < res.getColumnDimension(); i++) {
            sum += res.get(0,i);
        }
        for(int i = 0; i < res.getColumnDimension(); i++) {
            res.set(0,i, res.get(0,i)/sum);
        }
        return res;
    }


    private static Matrix sigmoid(Matrix input) {
        Matrix res = new Matrix(input.getRowDimension(), input.getColumnDimension());
        for(int i = 0; i < input.getRowDimension(); i++) {
            for(int j = 0; j < input.getColumnDimension(); j++) {
                res.set(i,j,1/(1+Math.exp(-input.get(i,j))));
            }
        }
        return res;
    }

    private static Matrix prelu(Matrix z, double a) {
        Matrix res = new Matrix(z.getRowDimension(), z.getColumnDimension());
        for(int i = 0; i < res.getRowDimension(); i++) {
            for(int j = 0; j < res.getColumnDimension(); j++) {
                double k = z.get(i,j);
                res.set(i,j,Math.max(0,k)+a*Math.min(0,k));
            }
        }
        return res;
    }

    private static Matrix relu(Matrix z) {
        return prelu(z, 0);
    }
}

