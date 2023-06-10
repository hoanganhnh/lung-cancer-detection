import cv2


def execute(path):

    # 1) IMAGE PROCESSING: Read and binarize image

    # Read scan image to OpenCV in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Binarize image using user-defined threshold
    bin_threshold = 100
    img_binary = cv2.threshold(img, bin_threshold, 255, cv2.THRESH_BINARY)[1]

    # 2) NOISE REMOVAL: Perform closing operation on image

    # Define an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Apply a closing operation to the image to reduce noise
    img_closed = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)

    # 3) SEGMENTATION: Segment tumor masses (if multiple) and find areas

    # Use processed image and identify tumor contours
    contours, _ = cv2.findContours(
        img_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # The maximum cross-sectional area of the tumor is a valuable marker to inform the clinician of tumor progression
    # Find the areas of all detected masses
    # conversion factor for px to cm for area, found based on PET scan dimensions
    conversion_factor = 314
    contour_areas = [round(cv2.contourArea(i)/conversion_factor, 3)
                     for i in contours]
    area = max(contour_areas)

    # Draw contours on image
    img_annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_annotated, contours, -1, (0, 0, 255), 2)

    # Save annotated scan as .jpg image to project directory
    cv2.imwrite('./static/images/annotated_scan.jpg', img_annotated)

    # Return annotated PET scan image and cross-sectional area
    return img_annotated, area
