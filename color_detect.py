import numpy as np
import cv2
import matplotlib.image as mpimg
import imutils
import matplotlib.pyplot as plt


def colour_detect(roi):
    roi_hsv = cv2.cvtColor(np.asarray(roi), cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 43, 46])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([156, 43, 46])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([124, 255, 255])
    lower_yellow = np.array([18, 43, 46])
    upper_yellow = np.array([34, 255, 255])

    mask_green = cv2.inRange(roi_hsv, lower_green, upper_green)
    mask_red1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
    mask_red = mask_red1 ^ mask_red2  # XOR
    mask_yellow = cv2.inRange(roi_hsv, lower_yellow, upper_yellow)

    roi_hsv_val = np.sum(roi_hsv)
    mask_green_val = np.sum(mask_green)
    mask_red_val = np.sum(mask_red)
    mask_yellow_val = np.sum(mask_yellow)

    if mask_green_val > mask_red_val and mask_green_val > mask_yellow_val and mask_green_val / roi_hsv_val > 0.045:
        colour = "Green "
        colour_mark = (0, 255, 0)
    elif mask_red_val > mask_green_val and mask_red_val > mask_yellow_val and mask_red_val / roi_hsv_val > 0.045:
        colour = "Red   "
        colour_mark = (255, 0, 0)
    elif mask_yellow_val > mask_green_val and mask_yellow_val > mask_red_val and mask_yellow_val / roi_hsv_val > 0.04:
        colour = "Yellow"
        colour_mark = (255, 255, 0)
    else:
        colour = "OFF   "
        colour_mark = (190, 190, 190)
    # print(mask_green_val, mask_red_val, mask_yellow_val)
    # print(colour, mask_green_val/roi_hsv_val, mask_red_val/roi_hsv_val, mask_yellow_val/roi_hsv_val)
    return colour, colour_mark


def find_contour(yolo_roi):
    # pre-process img: convert to bi-value img
    gray = cv2.cvtColor(yolo_roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Check for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    TL_regions = [] # coordinates

    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # filter some wrong contour
        if w > 5 and h > 5 and h / w > 1.8:
            TL_regions.append((x, y, w, h))

    TL_regions = sorted(TL_regions, key=lambda x:x[0]) # coordinates
    TLs = []
    for i in range(len(TL_regions)):
        x, y, w, h = TL_regions[i]
        TL_img = yolo_roi[y:y+h, x:x+w]
        TLs.append(TL_img)

    return TLs, TL_regions

"""
??merge???,??????????,????????(?w?h??)??????????????????
??,??????,????????,???????????,
????????,????????????????yolo?ROI???
"""

if __name__ == "__main__":
    image = mpimg.imread('./test_img/offset.jpg')
    col, colmark = colour_detect(image)
    print(col)

    tls, cors = find_contour(image)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    img2 = np.copy(image)
    for i in range(len(cors)):
        x, y, w, h = cors[i]
        cv2.rectangle(img2, (x, y+h), (x+w, y),(0, 0, 255), thickness=2)
    plt.imshow(img2)

