package net.tzolov.cv.cropper;

import java.util.ArrayList;
import java.util.List;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

public class Scanner {
    org.opencv.core.Mat srcBitmap;
    private boolean canny;
    public int resizeThreshold = 500;
    private float resizeScale = 1.0f;
    private boolean isHisEqual = false;


    public Scanner(org.opencv.core.Mat bgr, boolean canny) {
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        this.srcBitmap = bgr;
        this.canny = canny;
    }

    private org.opencv.core.Mat preprocessedImage(org.opencv.core.Mat image, int cannyValue, int blurValue) {
        org.opencv.core.Mat grayMat = new org.opencv.core.Mat();
        Imgproc.cvtColor(image, grayMat, Imgproc.COLOR_BGR2GRAY);
        if (!canny) {
            return grayMat;
        }

        if (isHisEqual){
            Imgproc.equalizeHist(grayMat, grayMat);
        }
        org.opencv.core.Mat blurMat = new org.opencv.core.Mat();
        Imgproc.GaussianBlur(grayMat, blurMat, new org.opencv.core.Size(blurValue, blurValue), 0);
        org.opencv.core.Mat cannyMat = new org.opencv.core.Mat();
        Imgproc.Canny(blurMat, cannyMat, (double) 50, (double) cannyValue, 3, false);
        org.opencv.core.Mat thresholdMat = new org.opencv.core.Mat();
        Imgproc.threshold(cannyMat, thresholdMat, 0, 255, Imgproc.THRESH_OTSU);
        return thresholdMat;
    }

    // indexing contours https://github.com/bytedeco/javacv/issues/1270#issuecomment-521129006
    List<Point2f> selectPoints(org.opencv.core.MatOfPoint2f contour) {
        List<org.opencv.core.Point> points = contour.toList();
        int size = points.size();
        List<Point2f> pts = new ArrayList<>();

        for(int i = 0; i < size; i++) {
            System.out.printf("size %d\n", size);
            System.out.printf("i %d\n", i);
            float x = (float) points.get(i).x;
            float y = (float) points.get(i).y;
            pts.add(new Point2f(x, y));
        }

        if (pts.size() > 4) {
            float x = pts.get(0).x();
            float y = pts.get(0).y();
            float minX = x;
            float maxX = x;
            float minY = y;
            float maxY = y;
            //得到一个矩形去包住所有点
            for (int i = 1; i < pts.size(); i++) {
                float pointX = pts.get(i).x();
                float pointY = pts.get(i).y();
                if (pointX < minX) {
                    minX = pointX;
                }
                if (pointX > maxX) {
                    maxX = pointX;
                }
                if (pointY < minY) {
                    minY = pointY;
                }
                if (pointY > maxY) {
                    maxY = pointY;
                }
            }

            //矩形中心点
            Point2f center = new Point2f((minX + maxX) / 2, (minY + maxY) / 2);
            //分别得出左上，左下，右上，右下四堆中的结果点
            Point2f p0 = choosePoint(center, pts, 0);
            Point2f p1 = choosePoint(center, pts, 1);
            Point2f p2 = choosePoint(center, pts, 2);
            Point2f p3 = choosePoint(center, pts, 3);
            pts.clear();
            //如果得到的点不是０，即是得到的结果点
            if (!(p0.x() == 0 && p0.y() == 0)){
                pts.add(p0);
            }
            if (!(p1.x() == 0 && p1.y() == 0)){
                pts.add(p1);
            }
            if (!(p2.x() == 0 && p2.y() == 0)){
                pts.add(p2);
            }
            if (!(p3.x() == 0 && p3.y() == 0)){
                pts.add(p3);
            }
        }
        return pts;
    }


    Point2f choosePoint(Point2f center, List<Point2f> points, int type) {
        int index = -1;
        int minDis = 0;

        if (type == 0) {
            for (int i = 0; i < points.size(); i++) {
                float x = points.get(i).x();
                float y = points.get(i).y();
                if (x < center.x() && y < center.y()) {
                    int dis = (int) (Math.sqrt(Math.pow((x - center.x()), 2) + Math.pow((y - center.y()), 2)));
                    if (dis > minDis){
                        index = i;
                        minDis = dis;
                    }
                }
            }
        } else if (type == 1) {
            for (int i = 0; i < points.size(); i++) {
                float x = points.get(i).x();
                float y = points.get(i).y();

                if (x < center.x() && y > center.y()) {
                    int dis = (int)(Math.sqrt(Math.pow((x - center.x()), 2) + Math.pow((y - center.y()), 2)));
                    if (dis > minDis){
                        index = i;
                        minDis = dis;
                    }
                }
            }
        } else if (type == 2) {
            for (int i = 0; i < points.size(); i++) {
                float x = points.get(i).x();
                float y = points.get(i).y();

                if (x > center.x() && y < center.y()) {
                    int dis = (int) (Math.sqrt(Math.pow((x - center.x()), 2) + Math.pow((y - center.y()), 2)));
                    if (dis > minDis){
                        index = i;
                        minDis = dis;
                    }
                }
            }

        } else if (type == 3) {
            for (int i = 0; i < points.size(); i++) {
                float x = points.get(i).x();
                float y = points.get(i).y();
                if (x > center.x() && y > center.y()) {
                    int dis = (int)(Math.sqrt(Math.pow((x - center.x()), 2) + Math.pow((y - center.y()), 2)));
                    if (dis > minDis){
                        index = i;
                        minDis = dis;
                    }
                }
            }
        }

        if (index != -1){
            return new Point2f(points.get(index).x(), points.get(index).y());
        }
        return new Point2f(0, 0);
    }

    public List<Point2f> scanPoint() {
        int cannyValue[] = {100, 150, 300};
        int blurValue[] = {3, 7, 11, 15};

        List<Point2f> result = new ArrayList<>();

        org.opencv.core.Mat image = resizeImage();
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 4; j++){

                org.opencv.core.Mat scanImage = preprocessedImage(image, cannyValue[i], blurValue[j]);
                List<MatOfPoint> contours = new ArrayList<>();
                org.opencv.core.Mat t = new org.opencv.core.Mat();
                Imgproc.findContours(scanImage, contours, t, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
                Collections.sort(contours, new Comparator<org.opencv.core.Mat>() {
                    public int compare(org.opencv.core.Mat mat1, org.opencv.core.Mat mat2) {
                        double v1Area = Math.abs(Imgproc.contourArea(mat1));
                        double v2Area = Math.abs(Imgproc.contourArea(mat2));
                        if (v1Area == v2Area)
                            return 0;
                        return v1Area > v2Area ? -1 : 1;
                    }
                });
                if (contours.size() > 0) {
                    org.opencv.core.MatOfPoint2f contour = new org.opencv.core.MatOfPoint2f(contours.get(0).toArray());
                    double arc = Imgproc.arcLength(new org.opencv.core.MatOfPoint2f(contour), true);
                    org.opencv.core.MatOfPoint2f outDP = new org.opencv.core.MatOfPoint2f();
                    //多变形逼近
                    Imgproc.approxPolyDP(contour, outDP, 0.01 * arc, true);
                    //筛选去除相近的点
                    List<Point2f> selectedPoints = selectPoints(outDP);
                    for(int k = 0; k < selectedPoints.size(); k++) {
                        System.out.printf("LOG: selectedPoints %d %f %f\n", k, selectedPoints.get(k).x(), selectedPoints.get(k).y());
                    }
                    if (selectedPoints.size() != 4) {
                        //如果筛选出来之后不是四边形
                        continue;
                    } else {
                        float widthMin = selectedPoints.get(0).x();
                        float widthMax = widthMin;
                        float heightMin = selectedPoints.get(0).y();
                        float heightMax = heightMin;
                        for (int k = 0; k < 4; k++) {
                            if (selectedPoints.get(k).x() < widthMin) {
                                widthMin = selectedPoints.get(k).x();
                            }
                            if (selectedPoints.get(k).x() > widthMax) {
                                widthMax = selectedPoints.get(k).x();
                            }
                            if (selectedPoints.get(k).y() < heightMin) {
                                heightMin = selectedPoints.get(k).y();
                            }
                            if (selectedPoints.get(k).y() > heightMax) {
                                heightMax = selectedPoints.get(k).y();
                            }
                        }
                        //选择区域外围矩形面积
                        float selectArea = (widthMax - widthMin) * (heightMax - heightMin);
                        int imageArea = scanImage.cols() * scanImage.rows();
                        if (selectArea < (imageArea / 20)) {
                            result.clear();
                            //筛选出来的区域太小
                            continue;
                        } else {
                            result = selectedPoints;
                            if (result.size() != 4) {
                                Point2f[] p = new Point2f[4];
                                p[0] = new Point2f(0, 0);
                                p[1] = new Point2f(image.cols(), 0);
                                p[2] = new Point2f(image.cols(), image.rows());
                                p[3] = new Point2f(0, image.rows());
                                result.add(p[0]);
                                result.add(p[1]);
                                result.add(p[2]);
                                result.add(p[3]);
                            }
                            for (Point2f p : result) {
                                p.x(p.x() * resizeScale);
                                p.y(p.y() * resizeScale);
                            }
                            // 按左上，右上，右下，左下排序
                            return sortPointClockwise(result);
                        }
                    }
                }
            }
        }
        //当没选出所需要区域时，如果还没做过直方图均衡化则尝试使用均衡化，但该操作只执行一次，若还无效，则判定为图片不能裁出有效区域，返回整张图
        if (!isHisEqual){
            isHisEqual = true;
            return scanPoint();
        }
        if (result.size() != 4) {
            Point2f[] p = new Point2f[4];
            p[0] = new Point2f(0, 0);
            p[1] = new Point2f(image.cols(), 0);
            p[2] = new Point2f(image.cols(), image.rows());
            p[3] = new Point2f(0, image.rows());
            result.add(p[0]);
            result.add(p[1]);
            result.add(p[2]);
            result.add(p[3]);
        }
        for (Point2f p : result) {
            p.x(p.x() * resizeScale);
            p.y(p.y() * resizeScale);
        }
        return sortPointClockwise(result);
    }


    List<Point2f> sortPointClockwise(List<Point2f> points) {
        if (points.size() != 4) {
            return points;
        }

        Point2f unFoundPoint = new Point2f();
        List<Point2f> result = new ArrayList<Point2f>();
        result.add(unFoundPoint);
        result.add(unFoundPoint);
        result.add(unFoundPoint);
        result.add(unFoundPoint);

        float minDistance = -1;
        for(Point2f point : points) {
            float distance = point.x() * point.x() + point.y() * point.y();
            if(minDistance == -1 || distance < minDistance) {
                result.set(0, point);
                minDistance = distance;
            }
        }
        if (result.get(0) != unFoundPoint) {
            Point2f leftTop = result.get(0);
            points.remove(leftTop);
            if ((pointSideLine(leftTop, points.get(0), points.get(1)) * pointSideLine(leftTop, points.get(0), points.get(2))) < 0) {
                result.set(2, points.get(0));
            } else if ((pointSideLine(leftTop, points.get(1), points.get(0)) * pointSideLine(leftTop, points.get(1), points.get(2))) < 0) {
                result.set(2, points.get(1));
            } else if ((pointSideLine(leftTop, points.get(2), points.get(0)) * pointSideLine(leftTop, points.get(2), points.get(1))) < 0) {
                result.set(2, points.get(2));
            }
        }
        if (result.get(0) != unFoundPoint && result.get(2) != unFoundPoint) {
            Point2f leftTop = result.get(0);
            Point2f rightBottom = result.get(2);
            points.remove(rightBottom);
            if (pointSideLine(leftTop, rightBottom, points.get(0)) > 0) {
                result.set(1, points.get(0));
                result.set(3, points.get(1));
            } else {
                result.set(1, points.get(1));
                result.set(3, points.get(0));
            }
        }

        if (result.get(0) != unFoundPoint && result.get(1) != unFoundPoint && result.get(2) != unFoundPoint && result.get(3) != unFoundPoint) {
            return result;
        }

        return points;
    }

    float pointSideLine(Point2f lineP1, Point2f lineP2, Point2f point) {
        float x1 = lineP1.x();
        float y1 = lineP1.y();
        float x2 = lineP2.x();
        float y2 = lineP2.y();
        float x = point.x();
        float y = point.y();
        return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1);
    }

    private org.opencv.core.Mat resizeImage() {
        int width = srcBitmap.cols();
        int height = srcBitmap.rows();

        int maxSize = width > height? width : height;
        if (maxSize > resizeThreshold) {
            resizeScale = 1.0f * maxSize / resizeThreshold;
            width = (int)(width / resizeScale);
            height = (int)(height / resizeScale);
            org.opencv.core.Size size = new org.opencv.core.Size(width, height);
            org.opencv.core.Mat resizedBitmap = new org.opencv.core.Mat(size, CvType.CV_8UC3);
            Imgproc.resize(srcBitmap, resizedBitmap, size);
            return resizedBitmap;
        }
        return srcBitmap;
    }
}
