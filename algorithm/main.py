import json
import sys

import torch

from .Classifier import Classifier
from .Segmenter import Segmenter
from .helper_functions import get_test_set, get_closest_split, process_single_staff_group, get_dir_content, \
    load_resnet101_model, gray, build_midi_file
from .preprocessing import *
from .staff import *
from .Coordinator import Coordinator


def main():
    input_folder, output_folder = sys.argv[1], sys.argv[2]
    files = get_dir_content(input_folder)
    images = "images"
    with open("utils/classes.json", "r") as fp:
        classes = json.load(fp)
    model = load_resnet101_model("ML/model.pt", len(classes))
    classes["invalid"] = -1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    classifier = Classifier(model, classes, device)
    with open("utils/note_to_pitch.json") as f:
        note_to_pitch = json.load(f)

    for file in files:
        print(f"processing file: {file}")
        original = cv2.imread(input_folder + "/" + file)
        original_gray = gray(original)
        cv2.imwrite(f"Test_{file}", original_gray)
        image = cv2.bitwise_not(original_gray)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        angle = skew_angle_hough_transform(image)
        if abs(angle) >= 1:
            #print(error)
            #angle += error
            print(angle)
            angle += angle / 10
            print(angle)
            image = rotate_image(image, angle)
            image = get_closer(image)

            original = rotate_image(original, angle)
            original_gray = rotate_image(original_gray, angle)

        image = get_binary_image(image)
        segmenter = Segmenter(image, original)
        cv2.imwrite(f"no_staffs_{file}", segmenter.image_without_staffs)

        boxes = segmenter.get_regions_of_interest()
        color = (0, 0, 255)
        image_no_staff = segmenter.image_without_staffs
        cv2.imwrite(f"temp/{file.split('.')[0]}_no_staffs.png", image_no_staff)
        copy = np.copy(original_gray)
        for box in boxes:
            box.draw_on_image(copy, color, 2)
        cv2.imwrite(f"boxes_{file}", copy)
        spacing, thickness = segmenter.spacing, segmenter.thickness
        first_staff = thickness[0]
        first_staff_thickness = first_staff[1] - first_staff[0]

        data_loader = get_test_set(original_gray, boxes)

        staff_centers = [(staff[1] + staff[0]) // 2 for staff in thickness]
        staffs_split = [[staff_centers[i:i+5] for i in range(0, len(staff_centers), 5)]]

        groups = get_closest_split(thickness, boxes, staff_centers)
        predictions = classifier.detect(data_loader)
        for i, prediction in enumerate(predictions):
            boxes[i].set_prediction(prediction)
        print(predictions)
        note_radius = spacing // 2 + first_staff_thickness
        num_groups = len(groups)
        notes = []
        time_step = 0
        if num_groups == 1:
            _, time_signature = process_single_staff_group(staff_centers, groups[0], spacing + first_staff_thickness, note_radius, notes, note_to_pitch, time_step)
        else:
            for i in range(0, num_groups - 1, 2):
                cur_time = time_step
                new_time, time_signature = process_single_staff_group(staffs_split[i], groups[i], spacing + first_staff_thickness, note_radius, notes, note_to_pitch, cur_time)
                if i < num_groups - 1:
                    _, time_signature2 = process_single_staff_group(staffs_split[i+1], groups[i+1], spacing + first_staff_thickness, note_radius, notes, note_to_pitch, cur_time)
                time_step = new_time
        for note in notes:
            print(note.get_name(), note.get_duration())
        output_file = build_midi_file(notes, time_signature)
        with open(f"{file.split('.')[0]}.midi", "wb") as f:
            output_file.writeFile(f)


def main2():
    input_folder, output_folder = sys.argv[1], sys.argv[2]
    files = get_dir_content(input_folder)
    with open("utils/classes.json", "r") as fp:
        classes = json.load(fp)
    model = load_resnet101_model("ML/model.pt", len(classes))
    classes["invalid"] = -1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    with open("utils/note_to_pitch.json") as f:
        note_to_pitch = json.load(f)

    coordinator = Coordinator(classes, model, device, note_to_pitch)

    for file in files:
        coordinator.process_image(file, input_folder, output_folder)


if __name__ == '__main__':
    main2()
