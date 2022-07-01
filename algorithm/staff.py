import numpy as np
from typing import List, Tuple
from collections import defaultdict
import cv2


MIN_PIXEL_VALUE, MAX_PIXEL_VALUE = 0, 255


def get_staffs(image: np.ndarray, skew=False):
    '''
    get staff lines in image.
    @param image: (np.ndarray) input image.
    @param skew: (bool) True if original image was skewed else false.
    @return: (list[int]) list of staff lines.
    '''
    rows, cols = image.shape
    staffs = []
    STAFF_THRESHOLD = 0.8
    SKEW_STAFF_THRESHOLD = 0.6

    for i in range(rows):
        row_sum = image[i].sum() // MAX_PIXEL_VALUE
        if row_sum >= STAFF_THRESHOLD * cols: # sum of row is above threshold
            staffs.append(i)
        elif skew and row_sum >= SKEW_STAFF_THRESHOLD * cols: # if image was skewed we allow lower threshold.
            staffs.append(i)
    return staffs


def calculate_thickness_and_spacing(staffs: List[int]):
    '''
    given list of staffs calculates staff line thickness and spacing between consecutive staff lines.
    @param staffs: (list[int]) staff lines.
    @return:
    '''
    if not staffs:
        return []
    thicknesses = []
    first = last = staffs[0]
    # calculate thickness of staff lines
    for i in range(1, len(staffs)):
        cur = staffs[i]
        if cur - 1 == last:
            last = cur
        else:
            thicknesses.append((first, last))
            first = last = cur
    thicknesses.append((first, last))
    # find most common thickness and then calculate spacing.
    result, most_common_spacing = most_common(thicknesses)

    return result, most_common_spacing


def most_common(found: List[Tuple[int, int]]):
    '''
    find most common thickness in staff lines,
    once discovered calculate most common spacing.
    @param found: (list[Tuple[int]]) list of staff lines thicknesses.
    @return: (Tuple[list[list[int]], int) return staff lines thickness array and most common spacing
    between staff lines according to said staff lines thickness.
    '''
    freqs = defaultdict(int)
    # calculate frequencies of thicknesses.
    for start, end in found:
        thickness = end - start
        freqs[thickness] += 1
    most_common_freq = 0
    # get most common thickness
    for freq in freqs:
        if freqs[freq] > most_common_freq:
            most_common_freq = freq
    result = []
    # get all staff lines with most common thickness.
    for start, end in found:
        thickness = end - start
        if thickness == most_common_freq:
            result.append([start, end])
        else:
            if thickness - 1 == most_common_freq or thickness + 1 == most_common_freq:
                result.append([start, end])

    spacings = defaultdict(int)
    last = result[0]
    # calculate spacing frequencies.
    for i in range(1, len(result)):
        cur = result[i]
        dis = cur[0] - last[1]
        last = cur
        spacings[dis] += 1

    # find most common spacing.
    most_common_spacing = None
    for key in spacings:
        if not most_common_spacing:
            most_common_spacing = key
        elif spacings[key] > most_common_spacing:
            most_common_spacing = key

    return result, most_common_spacing


def remove_staff_lines(image: np.ndarray):
    '''
    remove staff lines from image.
    @param image: (np.ndarray) input image.
    @return: (np.ndarray) output image.
    '''

    copy = np.copy(image)
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    REMOVE_KERNEL_SIZE = (100, 1) # best results
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, REMOVE_KERNEL_SIZE)
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    COLOR = (255, 255, 255) # Draw detected contours in white since image is in binary -> black and white.
    CNT_THICKNESS = 2
    DRAW_ALL_CONTOURS_FLAG = -1
    for c in cnts:
        cv2.drawContours(image, [c], DRAW_ALL_CONTOURS_FLAG, COLOR, CNT_THICKNESS)

    # Repair image
    REPAIR_KERNEL_SIZE = (1, 6) # best results
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, REPAIR_KERNEL_SIZE)
    result = MAX_PIXEL_VALUE - cv2.morphologyEx(MAX_PIXEL_VALUE - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    result = cv2.bitwise_not(result)

    return result

