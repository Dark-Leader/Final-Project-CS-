import cv2


class BoundingBox:
    '''
    represents a box which encapsulates a note or group of notes in a melody.
    '''
    def __init__(self, x, y, width, height, image=None, prediction=None):
        '''
        constructor
        @param x: (int) left border.
        @param y:  (int) top border.
        @param width: (int) width of the box in pixels.
        @param height: (int) height of the box in pixels.
        @param image: (np.ndaaray) sub image the box encapsulates
        @param prediction: classifier prediction for the box.
        '''
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = self.width * self.height
        self.center = (x + width // 2, y + height // 2)
        self.image = image
        self.prediction = prediction

    def overlap(self, other):
        '''
        check if two boxes are overlapping and by how much.
        @param other: (BoundingBox) other box.
        @return: (float) area they overlap.
        '''
        x = max(0, min(self.x + self.width, other.x + other.width) - max(self.x, other.x))
        y = max(0, min(self.y + self.height, other.y + other.height) - max(self.y, other.y))
        area = x * y
        return area / self.area

    def merge(self, other):
        '''
        merge two Bounding boxes into one.
        @param other: (BoundingBox) other box.
        @return: (BoundingBox) result box.
        '''
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        width = max(self.x + self.width, other.x + other.width) - x
        height = max(self.y + self.height, other.y + other.height) - y
        return BoundingBox(x, y, width, height)

    def draw_on_image(self, image, color, thickness):
        '''
        draw the box on the image.
        @param image: (np.ndarray) image to draw on.
        @param color: color the draw.
        @param thickness: thickness of strokes.
        @return: None.
        '''
        start = (int(self.x), int(self.y))
        end = (int(self.x + self.width), int(self.y + self.height))
        cv2.rectangle(image, start, end, color, thickness)

    def __str__(self):
        '''
        string representation of the box.
        @return: (str)
        '''
        return f"start_x = {self.x}, start_y = {self.y}, width = {self.width}, height = {self.height}"

    def set_img(self, img):
        '''
        setter for image.
        @param img: (np.ndarray) image the box encapsulates.
        @return: None.
        '''
        self.image = img

    def set_prediction(self, pred):
        '''
        setter for prediction.
        @param pred: (str) new prediction.
        @return: None.
        '''
        self.prediction = pred
