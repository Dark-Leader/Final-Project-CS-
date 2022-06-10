from collections import deque
import numpy as np

from algorithm.BoundingBox import BoundingBox
from algorithm.preprocessing import get_binary_image
from algorithm.staff import get_staffs, calculate_thickness_and_spacing, remove_staff_lines


class Segmenter:

    def __init__(self, bin_img: np.ndarray, original, skew=False):
        self.i = 0
        self.bin_img = bin_img
        self.rows, self.cols = bin_img.shape
        self.staffs = get_staffs(bin_img, skew)
        self.thickness, self.spacing = calculate_thickness_and_spacing(self.staffs)
        self.image_without_staffs = remove_staff_lines(original)

    def get_regions_of_interest(self):
        copy = np.copy(self.image_without_staffs)
        copy = get_binary_image(copy)
        rows, cols = copy.shape
        boxes = []
        queue = deque()
        visited = set()
        max_error = max(x[1] - x[0] for x in self.thickness) * 2.5
        print(max_error)
        offset = 2
        for row in range(rows):
            for col in range(cols):
                if copy[row][col] != 0:
                    queue.append((row, col, max_error))
                    min_x = max_x = col
                    min_y = max_y = row
                    while queue:
                        i, j, error = queue.popleft()
                        if copy[i][j] != 0:
                            min_x = min(min_x, j)
                            min_y = min(min_y, i)
                            max_x = max(max_x, j)
                            max_y = max(max_y, i)
                        copy[i][j] = 0
                        if i > 0 and (i - 1, j) not in visited:
                            if copy[i - 1][j] != 0:
                                queue.append((i - 1, j, max_error))
                                visited.add((i - 1, j))
                            elif error > 0:
                                queue.append((i - 1, j, error - 1))
                                visited.add((i - 1, j))
                        if i < rows - 1 and (i + 1, j) not in visited:
                            if copy[i + 1][j] != 0:
                                queue.append((i + 1, j, max_error))
                                visited.add((i + 1, j))
                            elif error > 0:
                                queue.append((i + 1, j, error - 1))
                                visited.add((i + 1, j))
                        if j > 0 and (i, j - 1) not in visited:
                            if copy[i][j - 1] != 0:
                                queue.append((i, j - 1, max_error))
                                visited.add((i, j - 1))
                            elif error > 0:
                                queue.append((i, j - 1, error - 1))
                                visited.add((i, j - 1))
                        if j < cols - 1 and (i, j + 1) not in visited:
                            if copy[i][j + 1] != 0:
                                queue.append((i, j + 1, max_error))
                                visited.add((i, j + 1))
                            elif error > 0:
                                queue.append((i, j + 1, error - 1))
                                visited.add((i, j + 1))
                    if max_x - min_x <= 2 and max_y - min_y <= 2:
                        continue
                    box = BoundingBox(min_x - offset, min_y - offset, max_x - min_x + 1 + offset, max_y - min_y + 1 + offset)
                    #print(box)
                    boxes.append(box)
        return boxes

    def dfs(self, image, i, j, box):
        rows, cols = image.shape
        if i < 0 or i >= rows or j < 0 or j >= cols or image[i][j] == 0:
            return
        image[i][j] = 0
        box[0] = min(box[0], j) # min_x of box
        box[1] = min(box[1], i) # min_y of box
        box[2] = max(box[2], j) # max_x of box
        box[3] = max(box[3], i) # max_y of box
        self.dfs(image, i+1, j, box)
        self.dfs(image, i-1, j, box)
        self.dfs(image, i, j+1, box)
        self.dfs(image, i, j-1, box)



