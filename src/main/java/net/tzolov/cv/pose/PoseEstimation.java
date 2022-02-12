package net.tzolov.cv.pose;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;

import org.bytedeco.javacpp.*;
import org.bytedeco.tensorflowlite.global.tensorflowlite.*;
import org.bytedeco.tensorflowlite.global.tensorflowlite;
import org.bytedeco.tensorflowlite.*;
import org.bytedeco.tensorflowlite.Interpreter;

import net.tzolov.cv.mtcnn.MtcnnService;
import net.tzolov.cv.mtcnn.MtcnnUtil;

import org.opencv.core.CvType;

public class PoseEstimation {
    public static final String TF_MODEL = "posenet/model.pb";
    public static final String TFLITE_MODEL = "/posenet/model.tflite";
    private FlatBufferModel tfliteModel;
    private Interpreter interpreter;

    private static String INPUT_NAME = "image";//"image";
    private static String HEATMAP = "Openpose/concat_stage7:0";// "MobilenetV1/heatmap_2/BiasAdd:0";
    private static String OFFSET = "MobilenetV1/offset_2/BiasAdd:0";
    private static String DISP_FORWARD = "MobilenetV1/displacement_fwd_2/BiasAdd:0";
    private static String DISP_BACKWARD = "MobilenetV1/displacement_bwd_2/BiasAdd:0";

    public PoseEstimation() {
        String model = getClass().getClassLoader().getResource(TFLITE_MODEL).getFile();
        tfliteModel = FlatBufferModel.BuildFromFile(model);
        TFLITE_MINIMAL_CHECK(tfliteModel != null && !tfliteModel.isNull());
        BuiltinOpResolver resolver = new BuiltinOpResolver();
        InterpreterBuilder builder = new InterpreterBuilder(tfliteModel, resolver);
        interpreter = new Interpreter((Pointer)null);

        builder.apply(interpreter);
        TFLITE_MINIMAL_CHECK(interpreter != null && !interpreter.isNull());

        TFLITE_MINIMAL_CHECK(interpreter.AllocateTensors() == tensorflowlite.kTfLiteOk);
    }

    static void TFLITE_MINIMAL_CHECK(boolean x) {
      if (!x) {
        System.err.print("Error at ");
        Thread.dumpStack();
        System.exit(1);
      }
    }

    public PoseResult poseDetection(BufferedImage img) throws IOException {
        //System.out.println("resizing image");
        BufferedImage copy = resizeImage(img, img.getWidth(), img.getHeight());

        img = Rotation.CLOCKWISE_90.rotate(img);
        img = resizeImage(img, 353, 257);

        TfLiteTensor inputImage = interpreter.input_tensor(0);
        TfLiteIntArray arr = interpreter.input_tensor(0).dims();

        FloatPointer p = interpreter.typed_input_tensor_float(0);

        int index = 0;
        int W = 353;
        int H = 257;
        int CHANNELS = 3;
        //[W][H][C]
        for(int i = 0; i < W; i++) {
            for(int j = 0; j < H; j++) {
                int rgb = img.getRGB(i, j);
                float r = (((rgb >> 16)  & 0xFF) - 128.f)/255.f;
                float g = (((rgb >> 8) & 0xFF) - 128.f)/255.f;
                float b = ((rgb & 0xFF) - 128.f)/255.f;
                p.put((H * CHANNELS * i) + (CHANNELS * j) + 0, r);
                p.put((H * CHANNELS * i) + (CHANNELS * j) + 1, g);
                p.put((H * CHANNELS * i) + (CHANNELS * j) + 2, b);
            }
        }

        TFLITE_MINIMAL_CHECK(interpreter.Invoke() == tensorflowlite.kTfLiteOk);

        TfLiteTensor data1 = interpreter.output_tensor(0);
        TfLiteTensor data2 = interpreter.output_tensor(1);
        TfLiteTensor data3 = interpreter.output_tensor(2);
        TfLiteTensor data4 = interpreter.output_tensor(3);

        TfLiteTensor[] t = {data1, data2, data3, data4};

        FloatPointer hmTensor = interpreter.typed_output_tensor_float(0);
        FloatPointer ofTensor = interpreter.typed_output_tensor_float(1);
        FloatPointer disp1Tensor = interpreter.typed_output_tensor_float(2);
        FloatPointer disp2Tensor = interpreter.typed_output_tensor_float(3);

        float hm[] = new float[1 * 45 * 33 * 17];
        float of[] = new float[1 * 45 * 33 * 34];
        float disp1[] = new float[1*45*33*32];
        float disp2[] = new float[1*45*33*32];

        hmTensor.get(hm, 0, hm.length);
        ofTensor.get(of, 0, of.length);
        disp1Tensor.get(disp1, 0, disp1.length);
        disp2Tensor.get(disp2, 0, disp2.length);

        INDArray rawHeatmaps = Nd4j.create(hm, new int[]{1, 45, 33, 17}); //data.get(HEATMAP);
        INDArray rawOffsets = Nd4j.create(of, new int[]{1, 45, 33, 34}); //data.get(OFFSET);

        Skeleton skeleton = new HumanSkeleton();

        long width = rawHeatmaps.shape()[2]; //33
        long height = rawHeatmaps.shape()[1]; //45
        float minPartThreshold = 0.3f;
        Dimension outputGridSize = new Dimension((int) width, (int) height);
        Dimension inputSize = new Dimension(257, 353);


        HeatmapScores heatmapScores = new HeatmapScores(rawHeatmaps, (int) outputGridSize.height, (int) outputGridSize.width, skeleton.getNumKeypoints());
        Offsets offsets = new Offsets(rawOffsets, (int) outputGridSize.height, (int) outputGridSize.width, skeleton.getNumKeypoints());

        PoseDecoder poseDecoder = new PoseDecoder(heatmapScores, offsets, outputGridSize, inputSize, minPartThreshold, skeleton);

        Pose pose = poseDecoder.decodePose();
        PoseResult res = new PoseResult(Arrays.asList(pose), 0.2f);
        return res;
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
