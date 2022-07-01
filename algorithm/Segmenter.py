from collections import deque
import numpy as np

from algorithm.BoundingBox import BoundingBox
from algorithm.preprocessing import get_binary_image
from algorithm.staff import get_staffs, calculate_thickness_and_spacing, remove_staff_lines
from config import helper


class Segmenter:
    '''
    segment input image into sections, then take each section and find notes in it -> save notes in a BoundingBox.
    '''
    def __init__(self, bin_img: np.ndarray, original, skew=False):
        '''
        constructor
        @param bin_img: (np.ndarray) input image.
        @param original: (np.ndarray) original RGB image.
        @param skew: (bool) was the input image skewed -> True or False.
        '''
        self.i = 0
        self.bin_img = bin_img
        self.rows, self.cols = bin_img.shape
        self.staffs = get_staffs(bin_img, skew) # get staff lines in the image.
        self.thickness, self.spacing = calculate_thickness_and_spacing(self.staffs) # calculate staff line thickness and spacing
        self.image_without_staffs = remove_staff_lines(original) # remove staff lines from image.

    def get_regions_of_interest(self):
        '''
        run BFS on the image with the staff lines, each pixel that is not 0 -> not black is a pixel of part of a note.
        we find the borders of the note -> top, right, left, bottom and encapsulate it in a BoundingBox.
        @return: (list[BoundingBox]) list of boxes of notes found.
        '''
        copy = np.copy(self.image_without_staffs)
        copy = get_binary_image(copy)
        rows, cols = copy.shape
        boxes = []
        queue = deque()
        visited = set()
        # since the process of removing staff lines is prone to removing parts of a note
        # we want to allow error of up to max_error in the BFS recursion.
        max_error = max(x[1] - x[0] for x in self.thickness) * 2.5
        offset = helper['allowed_offset'] # encapsulate in a box plus offset pixels margin of error.
        for row in range(rows):
            for col in range(cols):
                if copy[row][col] != 0: # not black pixel.
                    queue.append((row, col, max_error))
                    min_x = max_x = col
                    min_y = max_y = row
                    while queue:
                        i, j, error = queue.popleft()
                        if copy[i][j] != 0: # note black pixel.
                            # update borders of box.
                            min_x = min(min_x, j)
                            min_y = min(min_y, i)
                            max_x = max(max_x, j)
                            max_y = max(max_y, i)
                        copy[i][j] = 0
                        if i > 0 and (i - 1, j) not in visited: # top.
                            if copy[i - 1][j] != 0: # note black pixel.
                                queue.append((i - 1, j, max_error))
                                visited.add((i - 1, j))
                            elif error > 0: # allow error.
                                queue.append((i - 1, j, error - 1))
                                visited.add((i - 1, j))
                        if i < rows - 1 and (i + 1, j) not in visited: # bottom.
                            if copy[i + 1][j] != 0: # note black pixel.
                                queue.append((i + 1, j, max_error))
                                visited.add((i + 1, j))
                            elif error > 0: # allow error.
                                queue.append((i + 1, j, error - 1))
                                visited.add((i + 1, j))
                        if j > 0 and (i, j - 1) not in visited: # left.
                            if copy[i][j - 1] != 0: # not black pixel.
                                queue.append((i, j - 1, max_error))
                                visited.add((i, j - 1))
                            elif error > 0: # allow error.
                                queue.append((i, j - 1, error - 1))
                                visited.add((i, j - 1))
                        if j < cols - 1 and (i, j + 1) not in visited: # right
                            if copy[i][j + 1] != 0: # not black pixel.
                                queue.append((i, j + 1, max_error))
                                visited.add((i, j + 1))
                            elif error > 0: # allow error.
                                queue.append((i, j + 1, error - 1))
                                visited.add((i, j + 1))
                    if max_x - min_x <= 2 and max_y - min_y <= 2: # box is not too small.
                        continue
                    box = BoundingBox(min_x - offset, min_y - offset, max_x - min_x + 1 + offset, max_y - min_y + 1 + offset)
                    boxes.append(box)
        return boxes