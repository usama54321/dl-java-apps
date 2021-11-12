package net.tzolov.cv.pose;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.Dimension;
import java.awt.Graphics;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;

import java.awt.Canvas;
import net.tzolov.cv.mtcnn.MtcnnService;
import net.tzolov.cv.mtcnn.MtcnnUtil;

public class PoseEstimation {
    public static final String TF_MODEL = "posenet/model.pb";
    private static String INPUT_NAME = "image";
    private static String HEATMAP = "MobilenetV1/heatmap_2/BiasAdd:0";
    private static String OFFSET = "MobilenetV1/offset_2/BiasAdd:0";
    private static String DISP_FORWARD = "MobilenetV1/displacement_fwd_2/BiasAdd:0";
    private static String DISP_BACKWARD = "MobilenetV1/displacement_bwd_2/BiasAdd:0";

    private final GraphRunner modelRunner;
    private final Java2DNativeImageLoader imageLoader;

    public static void main(String[] args) {
        try {
            new PoseEstimation().test1();
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    void test1() throws IOException {
        String image = "/home/usama/ml_system/pose2.jpg";
        BufferedImage img = ImageIO.read(new File(image));
        Map<String, INDArray> data = poseDetection(img);

        INDArray rawHeatmaps = data.get(HEATMAP);
        INDArray rawOffsets = data.get(OFFSET);
        Skeleton skeleton = new HumanSkeleton();

        //[1, w, h, num_keypoints] -> [w, h, num_keypoints] -> [num_keypoints, w, h]
        MtcnnService.lg("rawOffsets", rawOffsets);
        MtcnnService.lg("rawHeatmaps", rawHeatmaps);
        rawHeatmaps = rawHeatmaps.get(point(0), all(), all(), all());//.permute(2,0,1);
        rawHeatmaps = Transforms.sigmoid(rawHeatmaps);
        MtcnnService.lg("rawHeatmaps", rawHeatmaps);
        rawOffsets = rawOffsets.get(point(0), all(), all(), all());//.permute(2,0,1);
        long width = rawHeatmaps.shape()[1];
        long height = rawHeatmaps.shape()[2];
        float minPartThreshold = 0.5f;
        Dimension outputGridSize = new Dimension((int) width, (int) height);
        Dimension inputSize = new Dimension(img.getWidth(), img.getHeight());

        HeatmapScores heatmapScores = new HeatmapScores(rawHeatmaps, (int) height, (int) width, skeleton.getNumKeypoints());
        Offsets offsets = new Offsets(rawOffsets, (int) height, (int) width, skeleton.getNumKeypoints());

        List<Pose> poses = new ArrayList<>();

        PoseDecoder poseDecoder = new PoseDecoder(heatmapScores, offsets, outputGridSize, inputSize, minPartThreshold, skeleton);

        Pose pose = poseDecoder.decodePose();

        if (pose != null) {
            poses.add(pose);
        }

        Graphics2D g = img.createGraphics();
        pose.draw(g, true, img.getWidth(), img.getHeight());
        ImageIO.write(img, "jpeg", new File("/home/usama/ml_system/java-app/mtcnn-java/test.jpeg"));
    }

    private static BufferedImage resizeImage(BufferedImage img, int width, int height) {
        Image image = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        Graphics2D g2d = resized.createGraphics();
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();

        return resized;
    }

    public PoseEstimation() {
        modelRunner = MtcnnUtil.createGraphRunner(
                getClass().getClassLoader().getResourceAsStream(TF_MODEL),
                PoseEstimation.INPUT_NAME,
                Arrays.asList(HEATMAP, OFFSET, DISP_FORWARD, DISP_BACKWARD)
                );
        imageLoader = new Java2DNativeImageLoader();
    }

    public Map<String, INDArray> poseDetection(INDArray img) {
        return this.modelRunner.run(Collections.singletonMap(PoseEstimation.INPUT_NAME, img));
    }

    public Map<String, INDArray> poseDetection(BufferedImage img) throws IOException {
        INDArray ndImage3HW = this.imageLoader.asMatrix(img).get(all(), all(), all(), all()).permutei(0,3,2,1);
        MtcnnService.lg("ndImage3HW", ndImage3HW);
		return poseDetection(ndImage3HW);
    }
}
