import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.feature import canny, corner_harris
from scipy.stats import mode

MAX_PIXEL_VALUE = 255

def get_binary_image(image: np.ndarray):
    '''
    convert grayscale image to binary by converting all pixels under threshold to 0
    and all pixels above threshold to 255.
    @param image: (np.ndarray) input image.
    @return: (np.ndarray) output image.
    '''
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.float64:
        image = image.astype('uint8')
    BINARY_THRESH = 128
    (thresh, binary_image) = cv2.threshold(image, BINARY_THRESH, MAX_PIXEL_VALUE, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_image


def check_is_horizontal(image: np.ndarray):
    '''
    check if image is not skewed by a finding staff line.
    if we managed to find a staff line then it's highly unlikely that the image is skewed.
    @param image: (np.ndarray) input image
    @return: (bool) true if image is not skewed else false.
    '''
    rows, cols = image.shape
    HORIZONTAL_THRESH = 0.2
    for i in range(rows):
        row_sum = image[i].sum() // MAX_PIXEL_VALUE
        if row_sum <= HORIZONTAL_THRESH * cols:
            return True
    return False


def skew_angle_hough_transform(image):
    '''
    calculate skew angle of image.
    @param image: (np.ndarray) input image.
    @return:
    '''
    # convert to edges
    edges = canny(image)
    # Classic straight-line Hough transform between 0.1 - 180 degrees.
    MIN_DEGREE = 0.1
    MAX_DEGREE = 180.0
    tested_angles = np.deg2rad(np.arange(MIN_DEGREE, MAX_DEGREE))
    h, theta, d = hough_line(edges, theta=tested_angles)

    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)

    # round the angles to 2 decimal places and find the most common angle.
    DECIMALS = 2
    most_common_angle = mode(np.around(angles, decimals=DECIMALS))[0]

    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi / 2)
    return skew_angle


def rotate_image(image, angle, scale=1.0):
    '''
    rotate image by 'angle' degrees.
    @param scale: (float) output image scale.
    @param image: (np.ndarray) input image.
    @param angle: (float) angle to rotate.
    @return: (np.ndarray) output image.
    '''
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def zoom_in(img: np.ndarray):
    '''
    remove border after image rotation.
    @param img: (np.ndarray) input image.
    @return: (np.ndarray) output image.
    '''
    rows, cols = img.shape
    start_good_row = 0
    # find first row that is not completely white.
    for i in range(rows):
        if img[i].sum() == 0:
            start_good_row = i
            break
    # find last row that is not completely white.
    end_good_row = rows - 1
    for j in range(rows - 1, -1, -1):
        if img[j].sum() == 0:
            end_good_row = j
            break

    start_good_col = 0
    end_good_col = cols - 1
    col_sums = img.sum(axis=0)
    # find first col that is not completely white.
    for col in range(cols):
        if col_sums[col] == 0:
            start_good_col = col
            break
    # find last col that is not completely white.
    for col in range(cols - 1, -1, -1):
        if col_sums[col] == 0:
            end_good_col = col
            break

    rows = list(range(start_good_row, end_good_row))
    cols = list(range(start_good_col, end_good_col))

    # create new image based of start_row, end_row, start_col, end_col
    to_select = np.ix_(rows, cols)
    new_img = img[to_select]
    return new_img


def get_closer(img):
    '''
    remove border after image rotation.
    @param img: (np.ndarray) input image.
    @return: (np.ndarray) output image.
    '''
    rows = []
    cols = []
    NUM_ROWS, NUM_COLS = img.shape[0], img.shape[1]
    INTERVAL = 16
    THRESH = 0.01
    for x in range(INTERVAL):
        no = 0
        for col in range(x * NUM_ROWS // INTERVAL, (x + 1) * NUM_ROWS // INTERVAL):
            for row in range(NUM_COLS):
                if img[col][row] == 0:
                    no += 1
        if no >= THRESH * NUM_COLS * NUM_ROWS // INTERVAL:
            rows.append(x * NUM_ROWS // INTERVAL)
    for x in range(INTERVAL):
        no = 0
        for row in range(x * NUM_COLS // INTERVAL, (x + 1) * NUM_COLS // INTERVAL):
            for col in range(NUM_ROWS):
                if img[col][row] == 0:
                    no += 1
        if no >= THRESH * NUM_ROWS * NUM_COLS // INTERVAL:
            cols.append(x * NUM_COLS // INTERVAL)
    new_img = img[rows[0]:min(NUM_ROWS, rows[-1] + NUM_ROWS // INTERVAL),
                  cols[0]:min(NUM_COLS, cols[-1] + NUM_COLS // INTERVAL)]
    return new_img
