package nn;

import org.imgscalr.Scalr;
import org.imgscalr.Scalr.Method;
import org.imgscalr.Scalr.Rotation;
import org.jblas.DoubleMatrix;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ImageLoader extends Loader {
    DoubleMatrix imgArr[][];
    DoubleMatrix lbls;

	int channels;
	int width, height;
    int[] counts;
	HashMap<String, Double> labelMap;
	ArrayList<String> names;


    public void ImageLoader(File folder, int channels, int width, int height) throws IOException {
        int z = 0;
        this.channels = channels;
        this.width = width;
        this.height = height;
        this.labelMap = constructLabelMap(folder);
        this.names = new ArrayList<String>();
        this.counts = countImages(folder, labelMap);
        int total = 0;
        for(int c : counts) {
            total += c;
        }
        this.lbls = new DoubleMatrix(total,labelMap.size());
        this.imgArr = new DoubleMatrix[total][3];

        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(File leaf : folder.listFiles()) {
            if(leaf.isDirectory() && labelMap.containsKey(leaf.getName())) {
                for(File image : leaf.listFiles()) {
                    Runnable ip = new ImageProcessor(image, labelMap.get(leaf.getName()), z);
                    executor.execute(ip);
                    z += 30;
                    z = z%total + z/total;
                }
            }
        }
        executor.shutdown();
        while(!executor.isTerminated());
    }

    public HashMap<String, Double> getLabelMap() {
        return labelMap;
    }

    private class ImageProcessor implements Runnable {
        private File image;
        private double leafNo;
        private int z;
        public ImageProcessor(File image, double leafNo, int z) {
            this.image = image;
            this.z = z;
            this.leafNo = leafNo;
        }

        public void run(){
            System.out.println(z);
            BufferedImage img = null;
            try {
                img = ImageIO.read(image);
            } catch (IOException e) {
                e.printStackTrace();
            }
            if ( (img.getHeight()-img.getWidth())*(height-width) < 0) {
                img = Scalr.rotate(img, Rotation.CW_90);
            }
            else {
                img = Scalr.rotate(img, Rotation.CW_180);
                img = Scalr.rotate(img, Rotation.CW_180);
            }
            img = Scalr.resize(img, Method.QUALITY, width, height);
            int[] pixels = ((DataBufferInt) img.getRaster().getDataBuffer()).getData();
            img.flush();
            for(int i = 0; i < channels; i++) {
                imgArr[z][i] = new DoubleMatrix(height, width);
            }
            if (pixels.length == width * height) {
                for (int i = 0; i < pixels.length; i++) {
                    for (int j = 0; j < channels; j++) {
                        imgArr[z][j].put(i/width, i%width, ((pixels[i] >>> (8 * j)) & 0xFF));
                    }
                }
                lbls.put(z, (int)leafNo, 1);

            }
        }
    }

    public DoubleMatrix[][] getTrainData(int i,  int batch) {
        DoubleMatrix[][] images = new DoubleMatrix[(imgArr.length-batch)][];
        int k = 0;
        for(int j = 0; j < imgArr.length; j++) {
            if(j < i*batch || j >= i*batch+batch) {
                images[k++] = imgArr[j];
            }
        }
        return Utils.normalizeData(images);
    }

    public DoubleMatrix[][] getTestData(int i, int batch) {
        DoubleMatrix[][] images = new DoubleMatrix[batch][];
        int k = 0;
        for(int j = 0; j < imgArr.length; j++) {
            if(j >= i*batch && j < i*batch+batch) {
                images[k++] = imgArr[j];
            }
        }
        return Utils.normalizeData(images);
    }

    private int[] countImages(File folder, HashMap<String, Double> labelMap) {
        int count[] = new int[labelMap.size()];
        for(File leaf : folder.listFiles()) {
            if (leaf.isDirectory() && labelMap.containsKey(leaf.getName())) {
                for (File image : leaf.listFiles()) {
                    count[labelMap.get(leaf.getName()).intValue()]++;
                }
            }
        }
        return count;
    }

	public ArrayList<String> getNames(File folder) {
		if(names == null) {
			names = new ArrayList<String>();
			for(File leaf : folder.listFiles()) {
				if(leaf.isDirectory()) {
                    for(File image : leaf.listFiles()) {
                        names.add(image.getName());
                    }
				}
			}
		}
		return names;
	}

    public static DoubleMatrix sample(final int patchRows, final int patchCols, final int numPatches, final DoubleMatrix[][] images) {
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
                    if(i%500 == 0)
                        System.out.println(threadNo+":"+i);
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
                    patches.putRow(threadNo*count+i, ch);
                }
            }
        }
        ExecutorService executor = Executors.newFixedThreadPool(Utils.NUMTHREADS);
        for(int i = 0; i < Utils.NUMTHREADS; i++) {
            if (i % 1000 == 0)
                System.out.println(i);
            Runnable patcher = new Patcher(i);
            executor.execute(patcher);
        }
        executor.shutdown();
        while(!executor.isTerminated());
        return patches;//normalizeData(patches);
    }

    public DoubleMatrix[][] getData() {
        return Utils.normalizeData(imgArr);
    }
    public DoubleMatrix getLabels() {
        return lbls;
    }

	public DoubleMatrix getTrainLabels(int i, int batch) {
        DoubleMatrix labs = null;
        for(int j = 0; j < lbls.rows; j++) {
            if(j < i*batch || j >= i*batch+batch) {
                if (labs == null) {
                    labs = lbls.getRow(j);
                }
                else {
                    labs = DoubleMatrix.concatVertically(labs, lbls.getRow(j));
                }
            }
        }
        return labs;
	}

    public DoubleMatrix getTestLabels(int i, int batch) {
        DoubleMatrix labs = null;
        for(int j = 0; j < lbls.rows; j++) {
            if(j >= i*batch && j < i*batch+batch) {
                if (labs == null) {
                    labs = lbls.getRow(j);
                }
                else {
                    labs = DoubleMatrix.concatVertically(labs, lbls.getRow(j));
                }
            }
        }
        return labs;
    }

	private HashMap<String, Double> constructLabelMap(File folder) {
		HashMap<String, Double> labelMap = new HashMap<String, Double>();
		double labelNo = -1;
		for(File leaf : folder.listFiles()) {
			labelNo++;
			leaf.listFiles();
			labelMap.put(leaf.getName(), labelNo);
		}
		return labelMap;
	}
}
