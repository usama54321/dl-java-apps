package net.tzolov.cv.pose;

import java.awt.Dimension;

import java.util.ArrayList;
import java.util.List;


/**
 * A developer-friendly class that contains the post-processed result from the PoseEstimation model.
 */
public class PoseResult {
    private List<Pose> poses;
    private float minPoseConfidence;

    public PoseResult(List<Pose> poses, float minPoseConfidence) { //, Dimension inputSize, Dimension sourceInputSize) {
        this.poses = poses;
        this.minPoseConfidence = minPoseConfidence;
    }

    /**
     * Get a list of poses returned from the model.
     *
     * @return a list of poses.
     */
    public List<Pose> getPoses() {
        return getPosesByThreshold(minPoseConfidence);
    }

    public List<Pose> getPosesByThreshold(float minConfidence) {
        List<Pose> resultPoses = new ArrayList<>();
        for (Pose pose : poses) {
            if (pose.getScore() >= minConfidence) {
                resultPoses.add(pose);
            }
        }

        return resultPoses;
    }
}
