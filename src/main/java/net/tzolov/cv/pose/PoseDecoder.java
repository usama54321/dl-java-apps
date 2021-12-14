package net.tzolov.cv.pose;

import java.awt.geom.Point2D;
import java.awt.Dimension;

public class PoseDecoder {

    private HeatmapScores heatmapScores;
    private Offsets offsets;
    private Dimension outputSize;
    private Dimension inputSize;
    private Skeleton skeleton;
    private float keypointThreshold;

    public PoseDecoder(HeatmapScores heatmapScores, Offsets offsets, Dimension outputSize, Dimension inputSize, float keypointThreshold, Skeleton skeleton) {
        this.heatmapScores = heatmapScores;
        this.offsets = offsets;
        this.outputSize = outputSize;
        this.inputSize = inputSize;
        this.skeleton = skeleton;
        this.keypointThreshold = keypointThreshold;
    }

    public Pose decodePose() {
        int numKeypoints = skeleton.getNumKeypoints();
        float[] maxScoresForParts = new float[numKeypoints];
        int[] maxRowIndex = new int[numKeypoints];
        int[] maxColIndex = new int[numKeypoints];


        int inputWidth = inputSize.width;
        int inputHeight = inputSize.height;

        int outputWidth = outputSize.width;
        int outputHeight = outputSize.height;

        for (int partId = 0; partId < numKeypoints; partId++) {
            for (int row = 0; row < outputHeight; row++) {
                for (int col = 0; col < outputWidth; col++) {
                    float score = heatmapScores.getScore(partId, col, row);
                    if (score > maxScoresForParts[partId]) {
                        maxScoresForParts[partId] = score;
                        maxRowIndex[partId] = row;
                        maxColIndex[partId] = col;
                    }
                }
            }
        }

        float scaleX = (float) inputWidth / outputWidth;
        float scaleY = (float) inputHeight / outputHeight;

        Keypoint[] partLocationOnImage = new Keypoint[numKeypoints];
        float totalKeypointScore = 0;
        for (int partId = 0; partId < numKeypoints; partId++) {
            int row = maxRowIndex[partId];
            int col = maxColIndex[partId];
            // Custom models stack offset values as [X, Y] which is the opposite of the pretrained
            // model. For now, all custom pose models use single pose decoding so we can assume
            // all are stacked with X first.
            Point2D.Float offsetPoint = offsets.getOffsetPoint(partId, col, row, true);

            float xLoc = col * scaleX + offsetPoint.x;
            float yLoc = row * scaleY + offsetPoint.y;

            partLocationOnImage[partId] = new Keypoint(partId, skeleton.getKeypointName(partId), new Point2D.Float(xLoc, yLoc), maxScoresForParts[partId], inputSize);
            totalKeypointScore += maxScoresForParts[partId];
        }


        // Use the average of the keypoint scores as the pose score.
        // Use the keypoint threshold to determine if we should show the keypoint on the pose.
        return new Pose(skeleton, partLocationOnImage, totalKeypointScore / maxScoresForParts.length, keypointThreshold, inputSize);
    }
}
