import cv2
import numpy as np

from algorithm.preprocessing import get_binary_image, rotate_image, skew_angle_hough_transform, get_closer
from algorithm.helper_functions import gray, get_test_set, process_single_staff_group, get_closest_split, build_midi_file
from algorithm.Classifier import Classifier
from algorithm.Segmenter import Segmenter


class Coordinator:

    def __init__(self, classes, model, device, note_to_pitch):
        self.classes = classes
        self.model = model
        self.classifier = Classifier(model, classes, device)
        self.note_to_pitch = note_to_pitch

    def process_image(self, file_name, input_folder, output_folder):
        print(f"processing file: {file_name}")
        original = cv2.imread(f"{input_folder}/{file_name}")
        copy = np.copy(original)
        original_gray = gray(original)
        #cv2.imwrite(f"Test_{file_name}", original_gray)
        image = cv2.bitwise_not(original_gray)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        angle = skew_angle_hough_transform(image)
        if abs(angle) >= 1: # attempt to correct the img skew angle by rotating the image
            angle += angle / 10
            image = rotate_image(image, angle)
            image = get_closer(image)
            original = rotate_image(original, angle)
            original_gray = rotate_image(original_gray, angle)

        image = get_binary_image(image)
        segmenter = Segmenter(image, original)
        #cv2.imwrite(f"no_staffs_{file}", segmenter.image_without_staffs)

        boxes = segmenter.get_regions_of_interest()
        #cv2.imwrite(f"temp/{file.split('.')[0]}_no_staffs.png", image_no_staff)
        #copy = np.copy(original_gray)
        #for box in boxes:
        #    box.draw_on_image(copy, color, 2)
        #cv2.imwrite(f"boxes_{file}", copy)
        spacing, thickness = segmenter.spacing, segmenter.thickness
        first_staff = thickness[0]
        first_staff_thickness = first_staff[1] - first_staff[0]

        data_loader = get_test_set(original_gray, boxes)

        staff_centers = [(staff[1] + staff[0]) // 2 for staff in thickness]
        staffs_split = [[staff_centers[i:i + 5] for i in range(0, len(staff_centers), 5)]]

        groups = get_closest_split(thickness, boxes, staff_centers)
        predictions = self.classifier.detect(data_loader)
        for i, prediction in enumerate(predictions):
            boxes[i].set_prediction(prediction)
        print(predictions)
        note_radius = spacing // 2 + first_staff_thickness
        num_groups = len(groups)
        notes = []
        time_step = 0
        all_centers = []
        if num_groups == 1:
            _, time_signature, centers = process_single_staff_group(staff_centers, groups[0], spacing + first_staff_thickness,
                                                           note_radius, notes, self.note_to_pitch, time_step)
            all_centers.append((centers, 0))
        else:
            for i in range(0, num_groups - 1, 2):
                cur_time = time_step
                new_time, time_signature, centers = process_single_staff_group(staffs_split[i], groups[i],
                                                                      spacing + first_staff_thickness, note_radius,
                                                                      notes, self.note_to_pitch, cur_time)
                all_centers.append((centers, i))
                if i < num_groups - 1:
                    _, time_signature2, centers = process_single_staff_group(staffs_split[i + 1], groups[i + 1],
                                                                    spacing + first_staff_thickness, note_radius, notes,
                                                                    self.note_to_pitch, cur_time)
                    all_centers.append((centers, i))
                time_step = new_time
        for note in notes:
            print(note.get_name(), note.get_duration())
        output_file = build_midi_file(notes, time_signature)

        for centers, idx in all_centers:
            try:
                copy = self.draw_centers(copy, centers, staff_centers[idx * 5 + 4], spacing)
            except IndexError:
                pass
        cv2.imwrite(f"{output_folder}/{file_name}_predictions.png", copy)

        with open(f"{output_folder}/{file_name.split('.')[0]}.midi", "wb") as f:
            output_file.writeFile(f)

    @staticmethod
    def draw_centers(image, centers, last_staff, spacing):
        print(centers)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 0, 0)
        thickness = 2
        for center in centers:
            x, y, name = center
            y_level = last_staff + spacing
            cv2.putText(image, name, (x, y_level), font, font_scale, color, thickness=thickness, lineType=cv2.LINE_AA)
        return image
