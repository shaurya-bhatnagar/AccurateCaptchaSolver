from PIL import Image
import numpy as np
import cv2


def clean(picture: Image) -> Image:
    picture = np.asarray(picture)[:, :, ::-1].copy()
    gray = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=3)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(picture, [c], -1, (255, 255, 255), 2)

    # get image spatial dimensions
    height, width = detected_lines.shape[:2]
    cv2.line(detected_lines, (width, 0), (0, height), (0, 0, 0), 2)

    # Dilate image
    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(detected_lines, kernel, iterations=1)

    # Convert image back to rgb
    dilated_image = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(dilated_image)

