package net.tzolov.cv.pose;

import org.apache.commons.lang3.tuple.ImmutablePair;

public class HumanSkeleton extends Skeleton {

    public static String[] PART_NAMES = {
            "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
            "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
            "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    };

    public static ImmutablePair[] CONNECTED_PART_NAMES = {
            new ImmutablePair<>("leftHip", "leftShoulder"),
            new ImmutablePair<>("leftElbow", "leftShoulder"),
            new ImmutablePair<>("leftElbow", "leftWrist"),
            new ImmutablePair<>("leftHip", "leftKnee"),
            new ImmutablePair<>("leftKnee", "leftAnkle"),
            new ImmutablePair<>("rightHip", "rightShoulder"),
            new ImmutablePair<>("rightElbow", "rightShoulder"),
            new ImmutablePair<>("rightElbow", "rightWrist"),
            new ImmutablePair<>("rightHip", "rightKnee"),
            new ImmutablePair<>("rightKnee", "rightAnkle"),
            new ImmutablePair<>("leftShoulder", "rightShoulder"),
            new ImmutablePair<>("leftHip", "rightHip")
    };

        public static ImmutablePair[] POSE_CHAIN = {
            new ImmutablePair<>("nose", "leftEye"),
            new ImmutablePair<>("leftEye", "leftEar"),
            new ImmutablePair<>("nose", "rightEye"),
            new ImmutablePair<>("rightEye", "rightEar"),
            new ImmutablePair<>("nose", "leftShoulder"),
            new ImmutablePair<>("leftShoulder", "leftElbow"),
            new ImmutablePair<>("leftElbow", "leftWrist"),
            new ImmutablePair<>("leftShoulder", "leftHip"),
            new ImmutablePair<>("leftHip", "leftKnee"),
            new ImmutablePair<>("leftKnee", "leftAnkle"),
            new ImmutablePair<>("nose", "rightShoulder"),
            new ImmutablePair<>("rightShoulder", "rightElbow"),
            new ImmutablePair<>("rightElbow", "rightWrist"),
            new ImmutablePair<>("rightShoulder", "rightHip"),
            new ImmutablePair<>("rightHip", "rightKnee"),
            new ImmutablePair<>("rightKnee", "rightAnkle")
    };

    public HumanSkeleton() {
        super("Human", PART_NAMES, CONNECTED_PART_NAMES, POSE_CHAIN);
    }
}
