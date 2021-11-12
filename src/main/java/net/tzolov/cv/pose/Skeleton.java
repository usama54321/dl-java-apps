package net.tzolov.cv.pose;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

public class Skeleton {

    protected String label;
    private String[] keypointNames;
    private Pair[] connectedKeypointNames;
    private Pair[] poseChain;

    private Map<String, Integer> keypointIds;
    private Pair[] connectedKeypointIndicies;
    private Pair[] parentChildIndicies;
    public static Integer[] parentToChildEdges;
    public static Integer[] childToParentEdges;

    public Skeleton(String label, String[] keypointNames) {
        this(label, keypointNames, new ImmutablePair[0], buildPoseChainFromKeypoints(keypointNames));
    }

    public Skeleton(String label, String[] keypointNames, Pair[] connectedKeypointNames, Pair[] poseChain) {
        this.label = label;
        this.keypointNames = keypointNames;
        this.connectedKeypointNames = connectedKeypointNames;
        this.poseChain = poseChain;

        this.keypointIds = new HashMap<>();
        for (int i = 0; i < keypointNames.length; i++) {
            String part = keypointNames[i];
            this.keypointIds.put(part, i);
        }

        this.connectedKeypointIndicies = new ImmutablePair[connectedKeypointNames.length];
        for (int i = 0; i < connectedKeypointNames.length; i++) {
            Pair<String, String> connectedPart = connectedKeypointNames[i];
            String jointA = connectedPart.getLeft();
            String jointB = connectedPart.getRight();
            connectedKeypointIndicies[i] = new ImmutablePair<>(keypointIds.get(jointA), keypointIds.get(jointB));
        }

        this.parentChildIndicies = new ImmutablePair[poseChain.length];
        for (int i = 0; i < poseChain.length; i++) {
            Pair<String, String> pose = poseChain[i];
            String jointA = pose.getLeft();
            String jointB = pose.getRight();
            parentChildIndicies[i] = new ImmutablePair<>(keypointIds.get(jointA), keypointIds.get(jointB));
        }

        this.parentToChildEdges = new Integer[parentChildIndicies.length];
        for (int i = 0; i < parentChildIndicies.length; i++) {
            Pair<Integer, Integer> parentChildEdge = parentChildIndicies[i];
            parentToChildEdges[i] = parentChildEdge.getRight();
        }

        this.childToParentEdges = new Integer[parentChildIndicies.length];
        for (int i = 0; i < parentChildIndicies.length; i++) {
            Pair<Integer, Integer> parentChildEdge = parentChildIndicies[i];
            childToParentEdges[i] = parentChildEdge.getLeft();
        }
    }

    public String getLabel() {
        return label;
    }

    public String[] getKeypointNames() {
        return keypointNames;
    }

    public int getNumKeypoints() {
        return keypointNames.length;
    }

    public String getKeypointName(int index) {
        return keypointNames[index];
    }

    public int getNumEdges() {
        return parentToChildEdges.length;
    }

    public Integer[] getChildToParentEdges() {
        return childToParentEdges;
    }

    public Integer[] getParentToChildEdges() {
        return parentToChildEdges;
    }

    public Pair[] getConnectedKeypointNames() {
        return connectedKeypointNames;
    }

    public Pair[] getConnectedKeypointIndicies() {
        return connectedKeypointIndicies;
    }

    public Pair[] getPoseChain() {
        return poseChain;
    }

    private static Pair[] buildPoseChainFromKeypoints(String[] keypointNames) {
        ImmutablePair[] poseChain = new ImmutablePair[keypointNames.length - 1];
        for (int kidx = 0; kidx < keypointNames.length - 1; kidx++) {
            poseChain[kidx] = new ImmutablePair<>(keypointNames[kidx], keypointNames[kidx + 1]);
        }
        return poseChain;
    }
}
