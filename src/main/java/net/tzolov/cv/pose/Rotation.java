package net.tzolov.cv.pose;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
/**
 * N.B. this example uses the new switch/case/Arrow notation, which requires Java 14.
 */
public enum Rotation {
    CLOCKWISE_90,
    CLOCKWISE_180,
    CLOCKWISE_270;

    public BufferedImage rotate(final BufferedImage original) {

        final int oW = original.getWidth();
        final int oH = original.getHeight();

        BufferedImage rotated = null;

        switch (this) {
            case CLOCKWISE_180:
                rotated = new BufferedImage(oW, oH, original.getType());
                break;
            default:
                rotated = new BufferedImage(oH, oW, original.getType());
                break;
        };

        final WritableRaster rasterOriginal = original.copyData(null);
        final WritableRaster rasterRotated  = rotated .copyData(null);
        /*
         * The Data for 1 Pixel...
         */
        final int[] onePixel = new int[original.getSampleModel().getNumBands()];
        /*
         * Copy the Pixels one-by-one into the result...
         */
        for (int x = 0; x < oW; x++) {
            for (int y = 0; y < oH; y++) {
                rasterOriginal.getPixel(x, y, onePixel);
                switch (this) {
                    case CLOCKWISE_90:
                        rasterRotated .setPixel(oH - 1 - y, x, onePixel);
                        break;
                    case CLOCKWISE_270:
                        rasterRotated .setPixel(y, oW - 1 - x, onePixel);
                        break;
                    default:
                        rasterRotated .setPixel(oW - 1 - x, oH - 1 - y, onePixel);
                };
            }
        }
        rotated.setData(rasterRotated);

        return rotated;
    }
}
