import joblib
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import (
    ball,
    disk,
    dilation,
    binary_erosion,
    remove_small_objects,
    erosion,
    closing,
    reconstruction,
    binary_closing,
    opening,
    binary_opening,
    binary_dilation,
)
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import cv2

knnModel = joblib.load("KNN.pkl")


def get_segmented_lungs(im, num, save=False, plot=False, crop_percentage=0.05):
    crop = im.copy()

    # Step 1: Crop the image
    height, width = im.shape[:2]
    start_row, start_col = int(height * 0.12), int(width * 0.12)
    end_row, end_col = int(height * 0.88), int(width * 0.88)
    crop = crop[start_row:end_row, start_col:end_col]

    # Step 2: Convert into a binary image.
    ret, binary = cv2.threshold(crop, 140, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Remove the blobs connected to the border of the image.
    cleared = clear_border(binary)

    # Step 4: Closure operation with a disk of radius 10. This operation is
    # to keep nodules attached to the lung wall.
    selem = disk(2)
    closing = binary_closing(cleared, selem)

    # Step 5: Label the image.
    label_image = label(closing)

    # Step 6: Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    segmented_area = label_image > 0

    # Step 7: seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    erosion = binary_erosion(segmented_area, selem)

    # Step 8: to keep nodules attached to the lung wall.
    selem = disk(10)
    closing2 = binary_closing(erosion, selem)

    # Step 9: Fill Holé
    edges = roberts(closing2)
    fill_holes = ndi.binary_fill_holes(edges)

    superimpose = crop.copy()
    get_high_vals = fill_holes == 0
    superimpose[get_high_vals] = 0

    superimpose = cv2.resize(superimpose, (512, 512))

    return superimpose


def predict(image):
    segment_lung = get_segmented_lungs(image)
    res = knnModel.predict(segment_lung)
    return res
