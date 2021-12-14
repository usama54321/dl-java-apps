package net.tzolov.cv.pose;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Wraps around the keypoint scores from the pose estimation model output.
 * @hide
 */
public class HeatmapScores {

    private INDArray scores;

    private int numKeypoints;
    private int height;
    private int width;

    public HeatmapScores(INDArray scores, int height, int width, int numKeypoints) {
        this.scores = scores;
        this.numKeypoints = numKeypoints;
        this.height = height;
        this.width = width;
    }

    public float getScore(int partId, int x, int y) {
        return scores.getFloat(0, y, x, partId);
    }

    public int getNumKeypoints() {
        return numKeypoints;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }
}
