import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import json
from typing import List
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from math import sqrt, pi, floor, ceil
from midiutil import MIDIFile

from algorithm.BoundingBox import BoundingBox
from algorithm.Note import Note
from algorithm.preprocessing import get_binary_image


beats_to_note = {"4": "whole", "2": "half", "1": "quarter", "0.5": "eighth", "0.25": "sixteenth"}
note_to_beats = {"whole": 4, "half": 2, "quarter": 1, "eighth": 0.5, "sixteenth": 0.25, "rest_whole": 4, "rest_half": 2, "rest_quarter": 1, "rest_eighth": 0.5, "rest_sixteenth": 0.25}


def get_note_locations(image, spacing, is_whole=False):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]

    # do morphology remove horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    lines1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    # do morphology to remove vertical lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    lines2 = cv2.morphologyEx(lines1, cv2.MORPH_CLOSE, kernel, iterations=1)
    lines2 = cv2.threshold(lines2, 128, 255, cv2.THRESH_BINARY)[1]

    # invert lines2
    lines2 = 255 - lines2

    # get contours
    cntrs = cv2.findContours(lines2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    circles = []

    # filter contours on area and draw good ones as black filled on white background
    for cntr in cntrs:
        area = cv2.contourArea(cntr)
        if spacing < area < (spacing * 2.2) ** 2:
            # get centroid
            M = cv2.moments(cntr)
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            #pt = "(" + str(cx) + "," + str(cy) + ")"
            # fit ellipse
            ellipse = cv2.fitEllipse(cntr)
            (x, y), (minor_axis, major_axis), angle = ellipse
            poly = cv2.ellipse2Poly((int(x), int(y)), (int(major_axis / 2), int(minor_axis / 2)), int(angle), 0, 360, 1)
            similarity = cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), cntr, cv2.CONTOURS_MATCH_I1,0)
            #print(similarity)
            if similarity < 0.35:
                center = (cx, cy)
                radius = sqrt(area / pi)
                if abs(radius - spacing) < 1:
                    circles.append((center, radius, True))
                elif abs(spacing / 3 - radius) < 1:
                    circles.append((center, radius, False))
                elif is_whole:
                    circles.append((center, radius, True))
    return circles


def get_test_set(original_image: np.ndarray, boxes: List[BoundingBox]):
    res = []
    offset = 2
    rows, cols = original_image.shape
    new_size = (224, 224)
    cv2.imwrite("temp/original.png", original_image)
    boxes.sort(key=lambda b: b.x)
    for i, box in enumerate(boxes):
        start_x, start_y = box.x, box.y
        end_x, end_y = start_x + box.width, start_y + box.height
        new_img = original_image[max(start_y - offset, 0): min(rows - 1, end_y + offset), max(start_x - offset, 0): min(cols - 1, end_x + offset)]
        box.set_img(np.copy(new_img))
        #cv2.imwrite(f"note_heads/{i+1}.png", new_img)
        #cv2.imwrite(f"temp/{i + 1}.png", new_img)
        bin_image = get_binary_image(new_img)
        scaled_image = cv2.resize(bin_image, new_size)
        #cv2.imwrite(f"scaled/{i + 1}.png", scaled_image)
        normalized_img = scaled_image / 255.0
        normalized_img = np.expand_dims(normalized_img, 0)
        res.append(normalized_img)
    test_x = torch.Tensor(np.asarray(res))
    test_y = torch.Tensor(np.asarray([np.array([0.]) for _ in range(len(res))]))
    tensor_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(tensor_dataset)

    return test_loader


def load_resnet101_model(path, num_classes):
    model = models.resnet101(False, (1, 224, 224), num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    return model


def get_dir_content(path):
    files = [os.fsdecode(file) for file in os.listdir(os.fsencode(path))]
    return files


def gray(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return new_image


def get_horizontal_image(image: np.ndarray):
    rows, cols = image.shape
    horizontal_size = cols // 20
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(image, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    horizontal = get_binary_image(horizontal)
    return horizontal


def get_vertical_image(image: np.ndarray):
    rows, cols = image.shape
    verticalsize = rows // 40
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(image, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    # Show extracted vertical lines

    vertical = get_binary_image(vertical)
    return vertical

def improve_img(image: np.ndarray):
    '''
            Extract edges and smooth image according to the logic
            1. extract edges
            2. dilate(edges)
            3. src.copyTo(smooth)
            4. blur smooth img
            5. smooth.copyTo(src, edges)
    '''
    # Step 1
    edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                  cv2.THRESH_BINARY, 3, -2)
    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel)
    # Step 3
    smooth = np.copy(image)
    # Step 4
    smooth = cv2.blur(smooth, (2, 2))
    # Step 5
    (rows, cols) = np.where(edges != 0)
    image[rows, cols] = smooth[rows, cols]
    return image


def get_closest_split(staffs, boxes: List[BoundingBox], staff_centers) -> List[List[BoundingBox]]:
    # each box is attached to its closest set of staffs
    num_staffs = len(staffs)
    arrays = [[] for _ in range(num_staffs // 5)]
    n = len(staff_centers)
    block_centers = staff_centers[2:n:5]
    #print(block_centers)
    for box in boxes:
        center = box.center
        x, y = center
        idx = -1
        min_dist = float("inf")
        for i, c in enumerate(block_centers):
            dist = abs(c - y)
            if dist < min_dist:
                min_dist = dist
                idx = i
        arrays[idx].append(box)
    return arrays


def fill_ellipse(image):
    copy = np.copy(image)
    if len(copy.shape) > 2:
        copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    des = cv2.bitwise_not(copy)
    _, contour, hier = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(des, [cnt], 0, 255, -1)
    res = cv2.bitwise_not(des)
    return res


def build_midi_file(notes: List[Note], time_signature):
    midi = MIDIFile(1)
    track = 0
    channel = 0
    time = 0  # In beats
    # duration = 1  # In beats
    tempo = 60  # In BPM
    volume = 100  # 0-127, as per the MIDI standard
    midi.addTempo(track, time, tempo)
    numerator, _ = time_signature.split("-")
    numerator = int(numerator)
    midi.addTimeSignature(track, time, numerator, 2, 24)
    notes.sort(key=lambda x: x.get_time())
    for note in notes:
        name, duration, pitch, time_step = note.get_name(), note.get_duration(), note.get_pitch(), note.get_time()
        if pitch == -1: # rest
            continue
        midi.addNote(track, channel, pitch, time_step, duration, volume)
    return midi


def add_to_history(intervals, interval, time_steps, time_step):
    intervals = insert_interval(intervals, interval)
    time_steps.append(time_step)
    return intervals


def process_single_staff_group(staffs, detections, spacing, note_radius, notes, name_to_pitch, time_step=0):
    clef = "sol_clef"
    last_note = None
    time_modifier = "4-4"
    centers = []
    for i, box in enumerate(detections):
        img = box.image
        prediction = box.prediction
        center_x, center_y = box.center
        x, y = box.x, box.y
        real_center_x, real_center_y = center_x + x, center_y + y
        is_whole = False
        if prediction == "invalid":
            continue
        if prediction in ["sol_clef", "fa_clef"]:
            clef = prediction
            centers.append((x, y, prediction))
            continue
        if prediction in ["4-4", "3-4", "2-4"]:
            time_modifier = prediction
            centers.append((x, y, prediction))
            continue
        if prediction == "dot":
            if last_note:
                time_step = dotted_note(last_note, time_step)
            continue
        if prediction == "barline":
            continue
        if prediction == "rest":
            min_dist = float("inf")
            idx = -1
            for j, staff in enumerate(staffs):
                dist = abs(staff - real_center_y)
                if dist < min_dist:
                    min_dist = dist
                    idx = j
            if staffs[idx] > real_center_y: # half rest
                name = "rest_half"
                duration = note_to_beats[name]
                note = Note(duration, name, -1, time_step)
                time_step += duration
                last_note = note
                notes.append(note)
                time_step += duration
            else:
                name = "rest_whole" # whole rest
                duration = note_to_beats[name]
                note = Note(duration, name, -1, time_step)
                time_step += duration
                last_note = note
                notes.append(note)
            continue
        if prediction == "rest_eighth":
            circles = get_note_locations(img, note_radius)
            duration = note_to_beats[prediction]
            num_circles = len(circles)
            if num_circles > 1:
                mult = 1.5 ** (num_circles - 1)
                duration *= mult
            duration = note_to_beats[prediction]
            note = Note(duration, prediction, -1, time_step)
            last_note = note
            notes.append(note)
            continue
        if prediction == "rest_quarter":
            circles = get_note_locations(img, note_radius)
            duration = note_to_beats[prediction]
            num_circles = len(circles)
            if num_circles > 0:
                mult = 1.5 ** num_circles
                duration *= mult
            duration = note_to_beats[prediction]
            note = Note(duration, prediction, -1, time_step)
            last_note = note
            notes.append(note)
            continue
        if prediction == "rest_sixteenth":
            circles = get_note_locations(img, note_radius)
            duration = note_to_beats[prediction]
            num_circles = len(circles)
            if num_circles > 2:
                mult = 1.5 ** (num_circles - 2)
                duration *= mult
            duration = note_to_beats[prediction]
            note = Note(duration, prediction, -1, time_step)
            last_note = note
            notes.append(note)
            continue
        if prediction == "whole":
            img = fill_ellipse(img)
            is_whole = True
        if prediction == "half":
            img = fill_ellipse(img)

        circles = get_note_locations(img, note_radius, is_whole)
        circles.sort(key=lambda c: c[0][0])
        duration = note_to_beats[prediction]
        for j, circle in enumerate(circles):
            center, radius, is_note = circle
            cur_x, cur_y = center
            cur_real_x, cur_real_y = ceil(x + cur_x), ceil(y + cur_y)
            cur_real_center = (cur_real_x, cur_real_y)
            if is_note:
                name = calculate_note(cur_real_center, clef, spacing, staffs)
                note = Note(duration, name, name_to_pitch[name], time_step)
                time_step += duration
                last_note = note
                notes.append(note)
                centers.append((cur_real_x - int(radius), cur_real_y, name))
            elif i > 0:
                time_step = dotted_note(last_note, time_step)

    return time_step, time_modifier, centers


def round_to_closest_half(num):
    return round(num * 2) / 2


def insert_interval(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    res = []
    i = 0
    n = len(intervals)
    start, end = new_interval
    intersection = -1
    while i < n and intervals[i][1] < start:
        res.append(intervals[i])
    while i < n and intervals[i][0] <= end:
        start, end = min(start, intervals[i][0]), max(end, intervals[i][1])
        if intersection != -1:
            intersection = i
        i += 1
    res.append([start, end])
    for k in range(i, n):
        res.append(intervals[k])
    return res, intersection


def dotted_note(note, time_step):
    duration = note.get_duration()
    addition = duration / 2
    note.set_duration(duration + addition)
    return time_step + addition


def calculate_note(center, clef, spacing, staffs):
    x, y = center
    note_cycle = "CDEFGAB"
    octave_size = len(note_cycle)
    octave = 4
    if clef == "sol_clef":
        base_height = staffs[-1] + spacing
    else:
        base_height = staffs[0] - spacing
    dist = base_height - y

    num_jumps = round_to_closest_half(dist / spacing)
    num_notes = floor(num_jumps * 2)
    abs_num_notes = abs(num_notes)
    if abs_num_notes >= octave_size:
        num_octaves = abs_num_notes // octave_size
        if num_notes < 0:
            num_octaves *= -1
            abs_num_notes += num_octaves * octave_size
        else:
            abs_num_notes -= num_octaves * octave_size
        octave += num_octaves

    if num_jumps < 0:
        octave -= 1
        note = note_cycle[-abs_num_notes]
    else:
        note = note_cycle[abs_num_notes]
    note += str(octave)
    return note


""" helper function to create all the music note audio files from 2nd octave to 6th """
def make_audio_files(output_folder):
    note_to_pitch = {}
    for speed in [0.25, 0.5, 1, 2, 4]:
        pitch = 95
        for octave in range(6, 1, -1):
            for name in ["B", "A#", "A", "G#", "G", "F#", "F", "E", "D#", "D", "C#", "C"]:
                if name + str(octave) not in note_to_pitch:
                    note_to_pitch[name + str(octave)] = pitch
                note = Note(speed, name + str(octave), pitch, 0)
                midi = build_midi_file([note])
                with open(f"{output_folder}/{note.get_name()}_{beats_to_note[note.get_duration()]}.mid", 'wb') as binfile:
                    midi.writeFile(binfile)
                pitch -= 1

    with open("note_to_pitch.json", "w") as fp:
        json.dump(note_to_pitch, fp)


