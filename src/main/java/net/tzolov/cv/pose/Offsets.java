package net.tzolov.cv.pose;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.awt.geom.Point2D;

/**
 * This class wraps around the offsets specified in the pose estimation model output.
 *
 * @hide
 */
public class Offsets {

    private INDArray rawOffsets;
    private int numParts;
    private int height;
    private int width;

    public Offsets(INDArray rawOffsets, int height, int width, int numParts) {
        this.rawOffsets = rawOffsets;
        this.numParts = numParts;
        this.height = height;
        this.width = width;
    }

    public int getNumParts() {
        return numParts;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public float getOffsetY(int partId, int x, int y) {
        //return rawOffsets.getFloat(4 * (y * width * numParts * 2 + x * numParts * 2 + partId));
        return rawOffsets.getFloat(0, y, x, partId);
    }

    public float getOffsetX(int partId, int x, int y) {
        return rawOffsets.getFloat(0, y, x, partId + numParts);
    }

    public Point2D.Float getOffsetPoint(int partId, int x, int y, boolean xFirst) {
        float offsetX = getOffsetX(partId, x, y);
        float offsetY = getOffsetY(partId, x, y);
        // If the offset matrix is stacked to be read [X, Y], switch the output order.
        if (xFirst) {
            return new Point2D.Float(offsetY, offsetX);
        }
        return new Point2D.Float(offsetX, offsetY);
    }

    public Point2D.Float getOffsetPoint(int partId, int x, int y) {
        return this.getOffsetPoint(partId, x, y, false);
    }
}
