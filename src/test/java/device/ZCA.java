package device;

import nn.ImageLoader;
import nn.Utils;
import org.jblas.DoubleMatrix;

import java.io.File;
import java.io.IOException;

/**
 * Created by jassmanntj on 6/7/2015.
 */
public class ZCA {
    public static void main(String[] args) throws IOException {
        int channels = 3;
        int imageColumns = 60;
        int imageRows = 80;
        File folder = new File("C:\\Users\\jassmanntj\\Desktop\\CA-Leaves2");
        ImageLoader loader = new ImageLoader(folder, channels, imageColumns, imageRows);
        DoubleMatrix[][] images = loader.getData();
        Utils.visualizeColorImg(images[0], "PREZCAL");
        images = Utils.ZCAWhiten(images, 1e-1);
        Utils.visualizeColorImg(images[0], "POSTZCAL");
        folder = new File("C:\\Users\\jassmanntj\\Desktop\\NotLeaves");
        loader = new ImageLoader(folder, channels, imageColumns, imageRows);
        images = loader.getData();
        Utils.visualizeColorImg(images[0], "PREZCA");
        images = Utils.ZCAWhiten(images, 1e-1);
        Utils.visualizeColorImg(images[0], "POSTZCA");

    }

}
