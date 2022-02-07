package net.tzolov.cv.cropper;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.tensorflowlite.BuiltinOpResolver;
import org.bytedeco.tensorflowlite.FlatBufferModel;
import org.bytedeco.tensorflowlite.Interpreter;
import org.bytedeco.tensorflowlite.InterpreterBuilder;
import org.bytedeco.tensorflowlite.TfLiteIntArray;
import org.bytedeco.tensorflowlite.global.tensorflowlite;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.tensorflow.conversion.TensorDataType;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.opencv.core.CvType;
import org.opencv.imgcodecs.Imgcodecs;

import net.tzolov.cv.pose.PoseEstimation;
import net.tzolov.cv.mtcnn.MtcnnUtil;

import org.bytedeco.opencv.opencv_core.Mat;

import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;


public class ImageDetector {
    public static int desiredSize = 256;
    private float THRESHOLD = 0.0f;

    //protected ByteBuffer imgData = null;
    //protected ByteBuffer outImgData = null;

    protected Interpreter interpreter;

    public ImageDetector() {}

    public synchronized org.opencv.core.Mat detectImage(org.opencv.core.Mat img, BufferedImage tmp) throws IOException {
        GraphRunner temp = MtcnnUtil.createGraphRunner(getClass().getClassLoader().getResourceAsStream("cropper/hed_graph.pb"), Arrays.asList("hed_input:0", "is_training:0"), Arrays.asList("hed/dsn_fuse/conv2d/BiasAdd:0"));
        INDArray image = new Java2DNativeImageLoader().asMatrix(tmp);//getClass().getClassLoader().getResourceAsStream(imageUri));

        image = image.div(255.).get(all(), interval(0,3), all(), all());

        Map<String, INDArray> data = new HashMap<>();
        data.put("hed_input:0", image.permutei(0, 2, 3, 1));
        data.put("is_training:0", Nd4j.scalar(false));
        Map<String, INDArray> out = temp.run(data);
        INDArray fin = null;
        for(Map.Entry<String, INDArray> kv: out.entrySet()) {
            fin = kv.getValue();
        }
        fin = fin.reshape(256, 256);

        org.opencv.core.Mat mat = new org.opencv.core.Mat(desiredSize, desiredSize, CvType.CV_8UC3);
        BufferedImage out1 = new BufferedImage(desiredSize, desiredSize, BufferedImage.TYPE_3BYTE_BGR);
        for(int i = 0; i < desiredSize; i++) {
            for (int j = 0; j < desiredSize; j++) {
                if (fin.getFloat(j, i) >= 0.5f) {
                    out1.setRGB(i, j, 0xFFFFFFFF);
                } else {
                    out1.setRGB(i, j, 0xFF000000);
                }
            }
        }

        byte[] data1 = ((DataBufferByte) out1.getRaster().getDataBuffer()).getData();
        mat.put(0, 0, data1);
        return mat;
    }
}
