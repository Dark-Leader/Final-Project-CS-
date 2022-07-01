import cv2
import numpy as np

from algorithm.preprocessing import get_binary_image, rotate_image, skew_angle_hough_transform, get_closer
from algorithm.helper_functions import gray, get_test_set, process_single_staff_group, get_closest_split, \
    build_midi_file, convert_midi_to_wav
from algorithm.Classifier import Classifier
from algorithm.Segmenter import Segmenter

MAX_PIXEL_VALUE = 255


class Coordinator:
    '''
    represents a class that coordinates all other classes in the algorithm portion of the project.
    '''

    def __init__(self, classes, model, device, note_to_pitch):
        '''
        constructor
        @param classes: (dict) dictionary of classes.
        @param model: (torch.nn.Module) trained model.
        @param device: (torch.device) cpu or gpu.
        @param note_to_pitch: (dict) dictionary that corresponds note name to pitch value.
        '''
        self.classes = classes
        self.model = model
        self.classifier = Classifier(model, classes, device)
        self.note_to_pitch = note_to_pitch

    def process_image(self, file_name, input_folder, output_folder, note_to_beats):
        '''
        process input image from user -> detect notes, make predictions and build output image, audio file.
        @param note_to_beats: (dict) dictionary of note name to number of beats.
        @param file_name: (str) input file name.
        @param input_folder: (str) path to input folder.
        @param output_folder: (str) path to output to save results.
        @return: (list[Note]) list of detected notes.
        '''
        # load image
        original = cv2.imread(f"{input_folder}/{file_name}")
        copy = np.copy(original)
        original_gray = gray(original)
        image = cv2.bitwise_not(original_gray)
        image = cv2.adaptiveThreshold(image, MAX_PIXEL_VALUE, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2) # best results

        angle = skew_angle_hough_transform(image)
        ANGLE_THRESHOLD = 1.0

        if abs(angle) >= ANGLE_THRESHOLD:  # attempt to correct the img skew angle by rotating the image
            ANGLE_CORRECTION = 1.1
            angle *= ANGLE_CORRECTION # best results -> the skew angle is usually off by around 10%.
            image = rotate_image(image, angle)
            image = get_closer(image)
            original = rotate_image(original, angle)
            original_gray = rotate_image(original_gray, angle)

        image = get_binary_image(image)
        segmenter = Segmenter(image, original)
        # get list of BoundingBoxes -> each box corresponds to a note or group of notes in the image.
        boxes = segmenter.get_regions_of_interest()

        spacing, thickness = segmenter.spacing, segmenter.thickness
        first_staff = thickness[0]
        first_staff_thickness = first_staff[1] - first_staff[0]

        data_loader = get_test_set(original_gray, boxes)  # create dataloader from list of boxes for classification.

        NUM_STAFFS_IN_BLOCK = 5
        staff_centers = [(staff[1] + staff[0]) // 2 for staff in thickness]
        staffs_split = [staff_centers[i:i + NUM_STAFFS_IN_BLOCK] for i in
                        range(0, len(staff_centers), NUM_STAFFS_IN_BLOCK)]  # split the staffs into groups of 5.

        # split the notes in the image to groups -> each note corresponds to the closest staff group
        groups = get_closest_split(thickness, boxes, staff_centers)
        predictions = self.classifier.detect(data_loader)  # make predictions on the notes.
        print(predictions)
        NON_ELLIPSE_NOTES = ["half", "whole"]
        for i, prediction in enumerate(predictions):
            boxes[i].set_prediction(prediction)
            if prediction in NON_ELLIPSE_NOTES:
                start_x, start_y = boxes[i].x, boxes[i].y
                end_x, end_y = start_x + boxes[i].width, start_y + boxes[i].height
                # cut original image and get sub matrix for the box.
                OFFSET = 10 # Best results if we expand the image by OFFSET pixels in each direction
                rows, cols = len(original_gray), len(original_gray[0])
                new_img = original_gray[max(start_y - OFFSET, 0): min(rows - 1, end_y + OFFSET),
                                        max(start_x - OFFSET, 0): min(cols - 1, end_x + OFFSET)]
                boxes[i].set_img(new_img)
                boxes[i].x -= OFFSET
                boxes[i].y -= OFFSET
        note_radius = spacing // 2 + first_staff_thickness  # expected radius of note in the image.
        num_groups = len(groups)
        notes = []  # list of detected notes.
        time_step = 0
        all_centers = []
        if num_groups == 1:  # simple case -> single group of 5 staffs
            # process the notes and calculate which note is being played at which timeStep.
            _, time_signature, centers = process_single_staff_group(staff_centers, groups[0],
                                                                    spacing + first_staff_thickness,
                                                                    note_radius, notes, self.note_to_pitch,
                                                                    note_to_beats, time_step)
            all_centers.append((centers, 0))
        else:
            # complex case -> more than one group of 5 staffs
            # split into groups of 2 consecutive staff groups.
            for i in range(0, num_groups, 2):
                # process the notes and calculate which note is being played at which timeStep.
                cur_time = time_step
                new_time, time_signature, centers = process_single_staff_group(staffs_split[i], groups[i],
                                                                               spacing + first_staff_thickness,
                                                                               note_radius,
                                                                               notes, self.note_to_pitch, note_to_beats,
                                                                               cur_time)
                all_centers.append((centers, i))
                if i < num_groups - 1:  # edge case to handle odd number of staff groups.
                    _, time_signature2, centers = process_single_staff_group(staffs_split[i + 1], groups[i + 1],
                                                                             spacing + first_staff_thickness,
                                                                             note_radius, notes,
                                                                             self.note_to_pitch, note_to_beats,
                                                                             cur_time)
                    all_centers.append((centers, i + 1))
                time_step = new_time
        for note in notes: # sanity check
            print(note.get_name(), note.get_duration())
        output_file = build_midi_file(notes)  # create output midi file from the list of notes.

        # for each note, draw its name on the output image.
        LAST_IDX_IN_GROUP = 4
        for centers, idx in all_centers:
            try:
                # draw prediction for each note in relation to each staff lines group
                copy = self.draw_centers(copy, centers, staff_centers[idx * NUM_STAFFS_IN_BLOCK + LAST_IDX_IN_GROUP], spacing)
            except IndexError:
                pass
        notes_only = [note for note in notes if (note.get_pitch() != -1)]  # remove rests.
        file_name_no_extension = file_name.split('.')[0]
        # save output image with the predictions drawn on it.
        output_file_name = f"{file_name_no_extension}_predictions.png"
        cv2.imwrite(f"{output_folder}/{output_file_name}", copy)

        MIDI_EXTENSION = ".midi"
        with open(f"{output_folder}/{file_name_no_extension}{MIDI_EXTENSION}", "wb") as f:
            output_file.writeFile(f)  # save midi file

        # convert midi to wav to play it on browser
        convert_midi_to_wav(output_folder, file_name_no_extension)

        return notes_only

    @staticmethod
    def draw_centers(image, centers, last_staff, spacing):
        '''
        draws note names on the output image.
        @param image: (np.ndarray) output image.
        @param centers: (list(Tuple)) list of centers of detected notes to draw.
        @param last_staff: (int) baseline for location of name drawing.
        @param spacing: (int) spacing between staff lines.
        @return: (np.ndarray) output image.
        '''
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.5
        COLOR = (MAX_PIXEL_VALUE, 0, 0)
        FONT_THICKNESS = 2
        for center in centers:
            x, y, name = center
            y_level = last_staff + spacing
            cv2.putText(image, name, (x, y_level), FONT, FONT_SCALE, COLOR, thickness=FONT_THICKNESS, lineType=cv2.LINE_AA)
        return image
