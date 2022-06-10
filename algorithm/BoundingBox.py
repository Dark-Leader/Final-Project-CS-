import cv2


class BoundingBox:
    def __init__(self, x, y, width, height, image=None, prediction=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = self.width * self.height
        self.center = (x + width // 2, y + height // 2)
        self.image = image
        self.prediction = prediction

    def overlap(self, other):
        x = max(0, min(self.x + self.width, other.x + other.width) - max(self.x, other.x))
        y = max(0, min(self.y + self.height, other.y + other.height) - max(self.y, other.y))
        area = x * y
        return area / self.area

    def merge(self, other):
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        width = max(self.x + self.width, other.x + other.width) - x
        height = max(self.y + self.height, other.y + other.height) - y
        return BoundingBox(x, y, width, height)

    def draw_on_image(self, image, color, thickness):
        start = (int(self.x), int(self.y))
        end = (int(self.x + self.width), int(self.y + self.height))
        cv2.rectangle(image, start, end, color, thickness)

    def __str__(self):
        return f"start_x = {self.x}, start_y = {self.y}, width = {self.width}, height = {self.height}"

    def set_img(self, img):
        self.image = img

    def set_prediction(self, pred):
        self.prediction = pred
