import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.feature import canny, corner_harris
from scipy.stats import mode


def get_binary_image(image: np.ndarray):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.float64:
        image = image.astype('uint8')
    (thresh, binary_image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_image


def check_is_horizontal(image: np.ndarray):
    rows, cols = image.shape
    for i in range(rows):
        row_sum = image[i].sum() // 255
        if row_sum <= 0.15 * cols:
            return True
    return False


def skew_angle_hough_transform(image):
    # convert to edges
    edges = canny(image)
    # Classic straight-line Hough transform between 0.1 - 180 degrees.
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)

    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)

    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2))[0]

    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi / 2)
    return skew_angle


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def zoom_in(img: np.ndarray):
    rows, cols = img.shape
    start_good_row = 0
    for i in range(rows):
        if img[i].sum() == 0:
            start_good_row = i
            break

    end_good_row = rows -1
    for j in range(rows - 1, -1, -1):
        if img[j].sum() == 0:
            end_good_row = j
            break

    start_good_col = 0
    end_good_col = cols -1
    col_sums = img.sum(axis=0)
    for col in range(cols):
        if col_sums[col] == 0:
            start_good_col = col
            break
    for col in range(cols -1, -1, -1):
        if col_sums[col] == 0:
            end_good_col = col
            break

    rows = list(range(start_good_row, end_good_row))
    cols = list(range(start_good_col, end_good_col))

    print(start_good_row, end_good_row)
    print(start_good_col, end_good_col)
    to_select = np.ix_(rows, cols)
    new_img = img[to_select]
    return new_img


def get_closer(img):
    rows = []
    cols = []
    for x in range(16):
        no = 0
        for col in range(x*img.shape[0]//16, (x+1)*img.shape[0]//16):
            for row in range(img.shape[1]):
                if img[col][row] == 0:
                    no += 1
        if no >= 0.01*img.shape[1]*img.shape[0]//16:
            rows.append(x*img.shape[0]//16)
    for x in range(16):
        no = 0
        for row in range(x*img.shape[1]//16, (x+1)*img.shape[1]//16):
            for col in range(img.shape[0]):
                if img[col][row] == 0:
                    no += 1
        if no >= 0.01*img.shape[0]*img.shape[1]//16:
            cols.append(x*img.shape[1]//16)
    new_img = img[rows[0]:min(img.shape[0], rows[-1]+img.shape[0]//16),
                  cols[0]:min(img.shape[1], cols[-1]+img.shape[1]//16)]
    return new_img





