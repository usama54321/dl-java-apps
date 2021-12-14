package net.tzolov.cv.cropper;

import java.awt.image.BufferedImage;
import java.io.File;
import java.awt.image.DataBufferByte;
import java.awt.Image;
import java.awt.Graphics2D;

import javax.imageio.ImageIO;

import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.opencv.core.CvType;



public class Utils {
    public static Mat bufferedImageToMat(BufferedImage image) {
        assert(image.getType() == BufferedImage.TYPE_INT_RGB);
        int w = image.getWidth(), h = image.getHeight();

        Size s = new Size(w, h);
        Mat res = new Mat(s, CvType.CV_8UC3);
        UByteIndexer indexer = res.createIndexer();
        for(int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                int rgb = image.getRGB(j, i);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                indexer.put(i, j, 2, r);
                indexer.put(i, j, 1, g);
                indexer.put(i, j, 0, b);
            }
        }
        indexer.release();

        return res;
    }

    public static void saveMatAsImage(Mat m, String path) {
        UByteIndexer indexer = m.createIndexer();       
        BufferedImage outBitmap = null;
        if (m.type() == CvType.CV_8UC1) {
            outBitmap = new BufferedImage(m.rows(), m.cols(), BufferedImage.TYPE_BYTE_GRAY);
        } else if (m.type() == CvType.CV_8UC3) {
            outBitmap = new BufferedImage(m.rows(), m.cols(), BufferedImage.TYPE_INT_RGB);
        } else if (m.type() == CvType.CV_8UC4) {
            outBitmap = new BufferedImage(m.rows(), m.cols(), BufferedImage.TYPE_INT_ARGB);
        }

        for(int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                if (m.type() == CvType.CV_8UC1) {
                    int color = indexer.get(i, j);
                    outBitmap.setRGB(i, j, color);
                } else if (m.type() == CvType.CV_8UC3) {
                    //assume bgr
                    int b = indexer.get(i, j, 0);
                    int g = indexer.get(i, j, 1);
                    int r = indexer.get(i, j, 2);
                    int color =  r << 16 | g << 8 | b;
                    outBitmap.setRGB(i, j, color);
                } else if (m.type() == CvType.CV_8UC4) {
                    //assume rgba
                    int r = indexer.get(i, j, 0);
                    int g = indexer.get(i, j, 1);
                    int b = indexer.get(i, j, 2);
                    int a = indexer.get(i, j, 3);
                    int color = a << 24 | r << 16 | g << 8 | b;
                    outBitmap.setRGB(i, j, color);
                }

            }
        }

        assert(outBitmap != null);
        try {
            assert(ImageIO.write(outBitmap, "jpeg", new File(path)));
        } catch (Exception e) {
            e.printStackTrace();
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static org.opencv.core.Mat threeByteBGRToMat(BufferedImage img) {
        assert(img.getType() == BufferedImage.TYPE_3BYTE_BGR);
        org.opencv.core.Mat ret = new org.opencv.core.Mat(img.getHeight(), img.getWidth(), CvType.CV_8UC3);
        byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        ret.put(0, 0, pixels);
        return ret;
    }

    public static BufferedImage resizeImage(BufferedImage img, int width, int height) {
        Image image = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();

        return resized;
    }
}
