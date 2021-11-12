package net.tzolov.cv.pose;

import java.awt.geom.Point2D;
import java.awt.Dimension;


/**
 * Keypoint indicating detected part on Pose.
 */
public class Keypoint {
    String name;
    private int id;
    private Point2D.Float position;
    private float score;
    private Dimension bounds;

    public Keypoint(int id, String name, Point2D.Float position, float score, Dimension bounds) {
        this.id = id;
        this.name = name;
        this.position = position;
        this.score = score;
        this.bounds = bounds;
    }

    public Point2D.Float getPosition() {
        return position;
    }

    public void setPosition(Point2D.Float position) {
        this.position = position;
    }

    public float getScore() {
        return score;
    }

    /**
     * The name of the keypoint.
     *
     * @return the name from the skeleton.
     */
    public String getName() {
        return name;
    }

    public int getId() {
        return id;
    }

    /**
     * Scale the keypoint for a given size.
     *
     * @param newBoundsSize - the bounds to scale the keypoint position for.
     * @return a new, scaled keypoint
     */
    public Keypoint scaled(Dimension newBoundsSize) {
        float scaleX = ((float) newBoundsSize.getWidth()) / (float) bounds.getWidth();
        float scaleY = ((float) newBoundsSize.getHeight()) / (float) bounds.getHeight();
        return new Keypoint(id, name, new Point2D.Float(position.x * scaleX, position.y * scaleY), score, newBoundsSize);
    }

    public float calculateSquaredDistanceFromCoordinates(Point2D.Float coordinates) {
        float dx = position.x - coordinates.x;
        float dy = position.y - coordinates.y;
        return dx * dx + dy * dy;
    }

    /*
    public JSONArray getPointsAsJsonArray() {
        JSONArray jsonArray = new JSONArray();
        try {
            jsonArray.put(id);
            jsonArray.put(position.x);
            jsonArray.put(position.y);
            jsonArray.put(score);
            return jsonArray;
        } catch (JSONException e) {
            throw new RuntimeException(e);
        }
    }
    */
}
