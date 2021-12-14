package net.tzolov.cv.pose;

import java.awt.Dimension;
import java.awt.Graphics;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

/**
 * A list of keypoints and an associated score.
 */
public class Pose {

    private Keypoint[] keypoints;
    private float poseScore;
    private float keypointThreshold;
    private Dimension bounds;
    private Skeleton skeleton;

    public Pose(Skeleton skeleton, Keypoint[] keypoints, float poseScore, float keypointThreshold, Dimension bounds) {
        this.keypoints = keypoints;
        this.poseScore = poseScore;
        this.keypointThreshold = keypointThreshold;
        this.bounds = bounds;
        this.skeleton = skeleton;
    }

    /**
     * Get all keypoints for this Pose.
     *
     * @return an array of keypoints.
     */
    public Keypoint[] getKeypoints() {
        return keypoints;
    }

    /**
     * Get the score of the Pose
     *
     * @return a float score from 0-1
     */
    public float getScore() {
        return poseScore;
    }

    /**
     * Get the keypoint threshold.
     *
     * @return the threshold.
     */
    public float getKeypointThreshold() {
        return keypointThreshold;
    }

    /**
     * Gets the bounds for the keypoint coordinates.
     *
     * @return the bounds.
     */
    public Dimension getBounds() {
        return bounds;
    }

    /**
     * Get the skeleton of the pose.
     *
     * @return the skeleton.
     */
    public Skeleton getSkeleton() {
        return skeleton;
    }

    /**
     * Create a new Pose with updated bounds. This will change the keypoint positions
     *
     * @param newBounds - the new bounds used to update the keypoint coordinates.
     * @return
     */
    public Pose scaledTo(Dimension newBounds) {
        return new Pose(skeleton, scaleKeypoints(keypoints, newBounds.width, newBounds.height), poseScore, keypointThreshold, bounds);
    }

    private Keypoint[] scaleKeypoints(Keypoint[] keypoints, int width, int height) {

        Keypoint[] updatedKeypoints = new Keypoint[keypoints.length];
        Dimension scaledSize = new Dimension(width, height);
        for (int i = 0; i < keypoints.length; i++) {
            Keypoint scaledKeypoints = keypoints[i].scaled(scaledSize);
            updatedKeypoints[i] = scaledKeypoints;
        }

        return updatedKeypoints;
    }

    /**
     * Get a list of the connected keypoints if the keypoint confidence scores are higher than a given threshold.
     *
     * @return a list of connected keypoints to draw a line between.
     */
    public List<Pair<Keypoint, Keypoint>> getConnectedKeypoints(Keypoint[] keypoints) {
        List<Pair<Keypoint, Keypoint>> connectedKeypoints = new ArrayList<>();
        for (Pair<Integer, Integer> connectedParts : skeleton.getConnectedKeypointIndicies()) {
            Keypoint leftKeypoint = keypoints[connectedParts.getLeft()];
            Keypoint rightKeypoint = keypoints[connectedParts.getRight()];

            if (leftKeypoint.getScore() >= keypointThreshold && rightKeypoint.getScore() >= keypointThreshold) {
                connectedKeypoints.add(new ImmutablePair<>(leftKeypoint, rightKeypoint));
            }
        }

        return connectedKeypoints;
    }

    /**
     * Draw the Pose on a canvas.
     *
     * @param canvas - the canvas to draw on.
     * @param connectParts - if true, draw lines connecting parts.
     */
    public void draw(Graphics graphics, boolean connectParts, int width, int height) {
        Keypoint[] updatedKeypoints = scaleKeypoints(keypoints, width, height);

        for (Keypoint keypoint : updatedKeypoints) {
            if(keypoint.getScore() >= 0.2f) {
                graphics.drawOval(Math.round(keypoint.getPosition().x), Math.round(keypoint.getPosition().y), 10, 10);
            }
        }
        if (connectParts) {

            for (Pair<Keypoint, Keypoint> connectedKeypoints : getConnectedKeypoints(updatedKeypoints)) {
                Keypoint left = connectedKeypoints.getLeft();
                Keypoint right = connectedKeypoints.getRight();
                graphics.drawLine(Math.round(left.getPosition().x), Math.round(left.getPosition().y), Math.round(right.getPosition().x), Math.round(right.getPosition().y));
            }
        }
    }

    /**
     * Draw the Pose on a canvas.
     *
     * @param canvas - the canvas to draw on.
     */
    public void draw(Graphics graphics, int width, int height) {
        draw(graphics, true, width, height);
    }

    /*
    public JSONArray getKeypointsAsJsonArray() {
        JSONArray keypointsArray = new JSONArray();
        for (Keypoint keypoint : keypoints) {
            keypointsArray.put(keypoint.getPointsAsJsonArray());
        }
        return keypointsArray;
    }
    */

    /*
    @Override
    public DataAnnotation toAnnotation( sourceInputSize) {
        float scaleX = ((float) sourceInput.getWidth()) / bounds.getWidth();
        float scaleY = ((float) sourceInput.getHeight()) / bounds.getHeight();

        List keypointAnnotations = new ArrayList();
        for (Keypoint keypoint : keypoints) {
            Keypoint scaledKeypoint = keypoint.scaled(sourceInput);
            keypointAnnotations.add(new KeypointAnnotation(
                    keypoint.getId(),
                    keypoint.getName(),
                    keypoint.getPosition().x * scaleX,
                    keypoint.getPosition().y * scaleY, true));
        }
        return new DataAnnotation(skeleton.getLabel(), keypointAnnotations, null, null,false);
    }
    */
}
