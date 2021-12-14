package net.tzolov.cv.cropper;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import java.awt.Point;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.opencv.core.CvType;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;

public class SmartCropper {
    public static ImageDetector sImageDetector = null;

    static {
        Loader.load(opencv_java.class);
    }

    public static void buildImageDetector(String modelFile) {
        sImageDetector = new ImageDetector();
    }

    public static Point2f[] getFullImgCropPoints(BufferedImage img) {
        Point2f[] data = new Point2f[] {null, null, null, null};
        data[0] = new Point2f(0,0);
        data[1] = new Point2f(img.getWidth(), 0);
        data[2] = new Point2f(img.getWidth(), img.getHeight());
        data[3] = new Point2f(0, img.getHeight());
        return data;
    }

    public static boolean canRightCrop(Point2f[] points) {
        Point2f lt = points[0];
        Point2f rt = points[1];
        Point2f rb = points[2];
        Point2f lb = points[3];
        return (pointSideLine(lt, rb, lb) * pointSideLine(lt, rb, rt) < 0) && (pointSideLine(lb, rt, lt) * pointSideLine(lb, rt, rb) < 0);
    }


    private static float pointSideLine(Point2f lineP1, Point2f lineP2, Point2f point) {
        return pointSideLine(lineP1, lineP2, point.x(), point.y());
    }

    private static float pointSideLine(Point2f lineP1, Point2f lineP2, float x, float y) {
        float x1 = lineP1.x();
        float y1 = lineP1.y();
        float x2 = lineP2.x();
        float y2 = lineP2.y();
        return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1);
    }

    public static BufferedImage cropImage(BufferedImage image) throws IOException {
        assert(image.getType() == BufferedImage.TYPE_3BYTE_BGR);
        org.opencv.core.Mat img = Utils.threeByteBGRToMat(image);
        org.opencv.core.Mat resized = new org.opencv.core.Mat(ImageDetector.desiredSize, ImageDetector.desiredSize, CvType.CV_8UC3);
        float scaleY = image.getHeight() / 256.f;
        float scaleX = image.getWidth() / 256.f;

        Imgproc.resize(img, resized, resized.size(), 0, 0, Imgproc.INTER_AREA);

        BufferedImage tmp = Utils.resizeImage(image, 256, 256);
        Point2f[] data = scan(resized, tmp);

        //scale points
        for(int i = 0; i < data.length; i++) {
            data[i].x(data[i].x() * scaleX);
            data[i].y(data[i].y() * scaleY);
        }

        if(!checkPoints(data)) {
            data = getFullImgCropPoints(image);
        }

        if (canRightCrop(data)) {
            org.opencv.core.Mat ret = SmartCropper.crop(img, data);
             MatOfByte matOfByte = new MatOfByte();
             Imgcodecs.imencode(".jpg", ret, matOfByte);
             //Storing the encoded Mat in a byte array
             byte[] byteArray = matOfByte.toArray();
             //Preparing the Buffered Image
             InputStream in = new ByteArrayInputStream(byteArray);
             BufferedImage cropped = ImageIO.read(in);
            return cropped;
        }

        return null;
    }

    public static org.opencv.core.Mat crop(org.opencv.core.Mat img, Point2f[] points) {
         if (img == null || points == null) {
            throw new IllegalArgumentException("srcBmp and cropPoints cannot be null");
        }
        if (points.length != 4) {
            throw new IllegalArgumentException("The length of cropPoints must be 4 , and sort by leftTop, rightTop, rightBottom, leftBottom");
        }

        Point2f leftTop = points[0];
        Point2f rightTop = points[1];
        Point2f rightBottom = points[2];
        Point2f leftBottom = points[3];

        int cropWidth = (int) ((CropUtils.getPointsDistance(leftTop, rightTop)
                + CropUtils.getPointsDistance(leftBottom, rightBottom))/2);
        int cropHeight = (int) ((CropUtils.getPointsDistance(leftTop, leftBottom)
                + CropUtils.getPointsDistance(rightTop, rightBottom))/2);

        return SmartCropper.nativeCrop(img, points, cropWidth, cropHeight);
    }

    private static org.opencv.core.Mat nativeCrop(org.opencv.core.Mat img, Point2f[] points, int newWidth, int newHeight) {
        if (points.length != 4) {
            return null;
        }
        Point2f leftTop = points[0];
        Point2f rightTop = points[1];
        Point2f rightBottom = points[2];
        Point2f leftBottom = points[3];

        assert(img.type() == CvType.CV_8UC3);

        org.opencv.core.Mat dstBitmapMatTwo = new org.opencv.core.Mat(newHeight, newWidth, img.type());

        //https://stackoverflow.com/questions/31184634/passing-point2f-as-arguments-for-getaffinetransform-in-javacv
        org.opencv.core.MatOfPoint2f src = new org.opencv.core.MatOfPoint2f(
                new org.opencv.core.Point(leftTop.x(), leftTop.y()),
                new org.opencv.core.Point(rightTop.x(), rightTop.y()),
                new org.opencv.core.Point(leftBottom.x(), leftBottom.y()),
                new org.opencv.core.Point(rightBottom.x(), rightBottom.y())
                );
        org.opencv.core.MatOfPoint2f dst = new org.opencv.core.MatOfPoint2f(
                new org.opencv.core.Point(0, 0),
                new org.opencv.core.Point(newWidth-1, 0),
                new org.opencv.core.Point(0, newHeight-1),
                new org.opencv.core.Point(newWidth-1, newHeight-1)
                );

        org.opencv.core.Mat transformTwo = Imgproc.getPerspectiveTransform(src, dst);

        Imgproc.warpPerspective(img, dstBitmapMatTwo, transformTwo, dstBitmapMatTwo.size());
        return dstBitmapMatTwo;
    }

    private static boolean checkPoints(Point2f[] points) {
        return points != null && points.length == 4
                && points[0] != null && points[1] != null && points[2] != null && points[3] != null;
    }

    public static Point2f[] scan(org.opencv.core.Mat image, BufferedImage temp) throws IOException {
        if (image == null)
            throw new IllegalArgumentException("image can not be null");

        org.opencv.core.Mat clone = image.clone();
        if (sImageDetector != null) {
            org.opencv.core.Mat bitmap = sImageDetector.detectImage(clone, temp);

            if (bitmap != null) {
                Imgproc.resize(bitmap, clone, new org.opencv.core.Size(image.width(), image.height()));
            }

        }

        Point2f[] outPoints = new Point2f[4];
        nativeScan(clone, outPoints, sImageDetector == null);
        return outPoints;
    }

    private static int[] rgb_to_bgr(int[] rgb) {
        int[] data = new int[rgb.length];
        for(int i = 0; i < rgb.length; i++) {
            int pixel = rgb[i];
            int r = (pixel >> 16 & 0xFF);
            int g = (pixel >> 8 & 0xFF);
            int b = (pixel & 0xFF);

            data[i] = (b << 16) | (g << 8) | (r << 0);
        }

        return data;
    }

    private static int[][] reshape2d(int[] data, int width, int height) {
        int[][] reshaped = new int[width][height];
        for(int i = 0; i < width; i++)
            for (int j = 0; j < height; j++)
                reshaped[i][j] = data[width * i + j];

        return reshaped;
    }

    private static void nativeScan(org.opencv.core.Mat srcImage, Point2f[] points, boolean canny) {
        assert(points.length == 4);
        if (points.length != 4) {
            return;
        }

        assert(srcImage.type() == CvType.CV_8UC3);
        //assume bgr image
        Scanner scanner = new Scanner(srcImage, canny);
        List<Point2f> scannedPoints = scanner.scanPoint();

        if(scannedPoints.size() == 4)
            for(int i = 0; i < 4; i++)
                points[i] = scannedPoints.get(i);

        return;
    }
}
