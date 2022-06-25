import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from typing import List
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from math import sqrt, pi, floor, ceil
from midiutil import MIDIFile


from algorithm.BoundingBox import BoundingBox
from algorithm.Note import Note
from algorithm.preprocessing import get_binary_image


def get_note_locations(image, spacing, is_whole=False):
    '''
    get note head locations -> ellipse shape of the note.
    @param image: (np.ndarray) input image.
    @param spacing: (int) spacing between two consecutive staff lines.
    @param is_whole: (bool) flag to check if current note is a whole note.
    @return: (list[Tuple]) list of detected note centers.
    '''
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

            if similarity < 0.35: # we only want shapes that are similar to an ellipse.
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
    '''
    build test set according to the provided image, list of boxes.
    @param original_image: (np.ndarray) input image.
    @param boxes: (list[BoundingBox]) list of boxes (detected notes).
    @return: (torch.utils.data.DataLoader) test set data loader.
    '''
    res = []
    offset = 2
    rows, cols = original_image.shape
    new_size = (224, 224) # model input size.

    boxes.sort(key=lambda b: b.x)
    for i, box in enumerate(boxes):
        start_x, start_y = box.x, box.y
        end_x, end_y = start_x + box.width, start_y + box.height
        # cut original image and get sub matrix for the box.
        new_img = original_image[max(start_y - offset, 0): min(rows - 1, end_y + offset), max(start_x - offset, 0): min(cols - 1, end_x + offset)]
        box.set_img(np.copy(new_img))

        bin_image = get_binary_image(new_img)
        scaled_image = cv2.resize(bin_image, new_size) # resize image to fit model input size.

        normalized_img = scaled_image / 255.0
        normalized_img = np.expand_dims(normalized_img, 0)
        res.append(normalized_img)
    # create data loader.
    test_x = torch.Tensor(np.asarray(res))
    test_y = torch.Tensor(np.asarray([np.array([0.]) for _ in range(len(res))]))
    tensor_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(tensor_dataset)

    return test_loader


def load_resnet101_model(path, num_classes):
    '''
    load resnet model from provided path.
    @param path: (str) path to model.pt file
    @param num_classes: number of classes.
    @return: (torch.nn.Module) output model.
    '''
    model = models.resnet101(False, (1, 224, 224), num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    checkpoint = torch.load(path)
    # update weights to saved weights.
    model.load_state_dict(checkpoint)
    return model


def get_dir_content(path):
    '''
    get directory content.
    @param path: (str) path to directory.
    @return: (list[str]) list of files in directory.
    '''
    files = [os.fsdecode(file) for file in os.listdir(os.fsencode(path))]
    return files


def gray(image):
    '''
    convert image to grayscale
    @param image: (np.ndarray) input image.
    @return: (np.ndarray) output image.
    '''
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return new_image


def get_horizontal_image(image: np.ndarray):
    '''
    get only horizontal lines (staff lines) from image.
    @param image: (np.ndarray) input image
    @return: (np.ndarray) output image.
    '''
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
    '''
    get only vertical lines (note lines, barlines) from image.
    @param image: (np.ndarray) input image
    @return: (np.ndarray) output image.
    '''
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
    @param image: (np.ndarray) input image
    @return: (np.ndarray) output image.
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
    '''
    split notes in image to groups -> each note goes to the closest staff lines group.
    @param staffs: (list[List[int]]) list of staff lines in image.
    @param boxes: (list[BoundingBox]) list of detected notes encapsulated by boxes.
    @param staff_centers: (list[int]) list of staff lines center y-axis value.
    @return: (list[list[BoundingBox]]) output list of list of boxes -> each box goes to the closest staff group.
    '''
    num_staffs = len(staffs)
    arrays = [[] for _ in range(num_staffs // 5)]
    n = len(staff_centers)
    block_centers = staff_centers[2:n:5] # get middle staff line in each staff line group.
    for box in boxes:
        center = box.center
        x, y = center
        idx = -1
        min_dist = float("inf")
        for i, c in enumerate(block_centers):
            dist = abs(c - y)
            if dist < min_dist: # if current staff group is closer to box.
                min_dist = dist
                idx = i
        arrays[idx].append(box)
    return arrays


def fill_ellipse(image):
    '''
    fill ellipse in image.
    @param image: (np.ndarray) input image.
    @return: (np.ndarray) output image.
    '''
    copy = np.copy(image)
    if len(copy.shape) > 2:
        copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    des = cv2.bitwise_not(copy)
    _, contour, hier = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(des, [cnt], 0, 255, -1)
    res = cv2.bitwise_not(des)
    return res


def build_midi_file(notes: List[Note]):
    '''
    build output midi file according to list detected notes.
    @param notes: (list[Note]) list of notes.
    @param time_signature: (str) time signature of the melody.
    @return: (MIDIFile) output midi file.
    '''
    midi = MIDIFile(1)
    track = 0
    channel = 0
    time = 0  # In beats
    # duration = 1  # In beats
    tempo = 60  # In BPM
    volume = 100  # 0-127, as per the MIDI standard
    midi.addTempo(track, time, tempo)

    notes.sort(key=lambda x: x.get_time())
    for note in notes:
        name, duration, pitch, time_step = note.get_name(), note.get_duration(), note.get_pitch(), note.get_time()
        if pitch == -1: # skip rest
            continue
        midi.addNote(track, channel, pitch, time_step, duration, volume)
    return midi


def add_to_history(intervals, interval, time_steps, time_step):
    '''
    add new interval to history of intervals.
    @param intervals: (list[list[int]]) list of previous intervals
    @param interval: (list[int]) current interval
    @param time_steps: (list[int]) previous timeSteps.
    @param time_step: (List[int]) current timeStep.
    @return:
    '''
    intervals = insert_interval(intervals, interval)
    time_steps.append(time_step)
    return intervals


def process_single_staff_group(staffs, detections, spacing, note_radius, notes, name_to_pitch, note_to_beats, time_step=0):
    '''
    precoess single group of notes associated with a single group of staff lines.
    @param staffs: (list[int]) list of current staff lines.
    @param detections: (list[str)) list of predictions.
    @param spacing: (int) spacing between two staff lines.
    @param note_radius: (int) expected note radius.
    @param notes: (list[Note]) list of previous notes.
    @param name_to_pitch: (dict) dictionary of note name to pitch value.
    @param note_to_beats: (dict) dictionary of note name to number of beats.
    @param time_step: (int) current timeStep in the melody.
    @return: (Tuple[int, str, list[list[int]]]) new timestep, time signature, list of detected note centers.
    '''
    found_clef = found_time_signature = False
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
        if prediction == "invalid": # invalid box.
            continue
        if prediction in ["sol_clef", "fa_clef"]: # found clef
            clef = prediction
            found_clef = True
            centers.append((x, y, prediction))
            continue
        if prediction in ["4-4", "3-4", "2-4"]: # found time signature.
            time_modifier = prediction
            found_time_signature = True
            centers.append((x, y, prediction))
            continue
        if not found_clef or not found_time_signature:
            continue
        if prediction == "dot": # found dot
            if last_note: # if we already saw a note then it is a dotted note.
                time_step = dotted_note(last_note, time_step)
            continue
        if prediction == "barline": # found barline.
            continue
        if prediction == "rest": # found half rest or whole rest -> need to check.
            min_dist = float("inf")
            idx = -1
            for j, staff in enumerate(staffs):
                dist = abs(staff - real_center_y)
                if dist < min_dist:
                    min_dist = dist
                    idx = j
            if staffs[idx] > real_center_y: # half rest -> note is above closest staff line.
                name = "rest_half"
                duration = note_to_beats[name]
                note = Note(duration, name, -1, time_step)
                time_step += duration
                last_note = note
                notes.append(note)
                time_step += duration
            else:
                name = "rest_whole" # whole rest -> note is below closest staff line.
                duration = note_to_beats[name]
                note = Note(duration, name, -1, time_step)
                time_step += duration
                last_note = note
                notes.append(note)
            continue
        if prediction == "rest_eighth": # found rest eighth.
            circles = get_note_locations(img, note_radius)
            duration = note_to_beats[prediction]
            num_circles = len(circles)
            if num_circles > 1: # dotted rest -> need to increase duration.
                mult = 1.5 ** (num_circles - 1)
                duration *= mult
            duration = note_to_beats[prediction]
            note = Note(duration, prediction, -1, time_step)
            time_step += duration
            last_note = note
            notes.append(note)
            continue
        if prediction == "rest_quarter": # found rest quarter.
            circles = get_note_locations(img, note_radius)
            duration = note_to_beats[prediction]
            num_circles = len(circles)
            if num_circles > 0: # dotted rest -> need to increase duration.
                mult = 1.5 ** num_circles
                duration *= mult
            duration = note_to_beats[prediction]
            note = Note(duration, prediction, -1, time_step)
            time_step += duration
            last_note = note
            notes.append(note)
            continue
        if prediction == "rest_sixteenth": # found rest sixteenth
            circles = get_note_locations(img, note_radius)
            duration = note_to_beats[prediction]
            num_circles = len(circles)
            if num_circles > 2: # dotted rest -> need to increase duration.
                mult = 1.5 ** (num_circles - 2)
                duration *= mult
            duration = note_to_beats[prediction]
            note = Note(duration, prediction, -1, time_step)
            time_step += duration
            last_note = note
            notes.append(note)
            continue
        if prediction == "whole": # found whole note.
            img = fill_ellipse(img) # fill the ellipse to get better results.
            is_whole = True
        if prediction == "half": # found half note.
            img = fill_ellipse(img) # fill the ellipse to get better results.
        # if we got here -> then the note is whole, half, quarter, eighth, sixteenth
        # need to check duration and find note heads to decide which piano key represents the note.
        circles = get_note_locations(img, note_radius, is_whole)
        circles.sort(key=lambda c: c[0][0])
        duration = note_to_beats[prediction]
        for j, circle in enumerate(circles): # iterate over note heads
            center, radius, is_note = circle
            cur_x, cur_y = center
            cur_real_x, cur_real_y = ceil(x + cur_x), ceil(y + cur_y)
            cur_real_center = (cur_real_x, cur_real_y)
            if is_note: # if it is a note.
                name = calculate_note(cur_real_center, clef, spacing, staffs)
                note = Note(duration, name, name_to_pitch[name], time_step)
                time_step += duration
                last_note = note
                notes.append(note)
                centers.append((cur_real_x - int(radius), cur_real_y, name))
            elif i > 0: # it is a dot -> so last note is a dotted note.
                if last_note:
                    time_step = dotted_note(last_note, time_step)

    return time_step, time_modifier, centers


def round_to_closest_half(num):
    '''
    round number to closest half -> if number is x.y
    then if y <= 0.25 then the result is x.
    if 0.25 < y <= 0.75 return x.5
    else return x + 1.
    @param num: (int) number to round.
    @return: (int) closest round to half.
    '''
    return round(num * 2) / 2


def insert_interval(intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
    '''
    insert new interval into list of intervals
    @param intervals: (list[list[int]]) list of previous intervals.
    @param new_interval: (list[int]) new interval.
    @return: (list[list[int]]) updated list of intervals.
    '''
    res = []
    i = 0
    n = len(intervals)
    start, end = new_interval
    intersection = -1
    while i < n and intervals[i][1] < start: # while current interval happens after start of previous intervals
        res.append(intervals[i])
    while i < n and intervals[i][0] <= end: # get the intersection of the intervals in the array.
        start, end = min(start, intervals[i][0]), max(end, intervals[i][1])
        if intersection != -1:
            intersection = i
        i += 1
    res.append([start, end]) # append intersection
    for k in range(i, n): # append intervals after intersection.
        res.append(intervals[k])
    return res, intersection


def dotted_note(note, time_step):
    '''
    if a note is a dotted note then its duration needs to be increased by 50%.
    and because the last note is a dotted note -> we need to update current timeStep of the melody.
    @param note: (Note) last note.
    @param time_step: (int) current timeStep of the melody.
    @return: (int) new timeStep of the melody.
    '''
    duration = note.get_duration()
    addition = duration / 2
    new_duration = duration + addition
    if new_duration > 3:
        return time_step
    note.set_duration(new_duration)
    return time_step + addition


def calculate_note(center, clef, spacing, staffs):
    '''
    given a note head center and a clef -> determine which note in the piano needs to be played.
    @param center: (list[int]) center of note.
    @param clef: (str) clef of the melody.
    @param spacing: (int) spacing between consecutive staff lines.
    @param staffs: (list[int]) current group staff lines.
    @return: (str) piano key name.
    '''
    x, y = center
    note_cycle = "CDEFGAB"
    octave_size = len(note_cycle)
    octave = 4 # base octave is 4 -> center of the piano.
    if clef == "sol_clef":
        base_height = staffs[-1] + spacing # C4 is below the last staff line.
    else:
        base_height = staffs[0] - spacing # C4 is above the top staff line.
    dist = base_height - y

    num_jumps = round_to_closest_half(dist / spacing) # number of staff lines spacings.
    num_notes = floor(num_jumps * 2) # between two staff lines there are 2 keys.
    abs_num_notes = abs(num_notes)
    if abs_num_notes >= octave_size: # need to jump more than a full octave.
        num_octaves = abs_num_notes // octave_size # num of octaves to jump.
        # jump remaining key count according to clef and number of notes left.
        if num_notes < 0:
            num_octaves *= -1
            abs_num_notes += num_octaves * octave_size
        else:
            abs_num_notes -= num_octaves * octave_size
        octave += num_octaves

    if num_jumps < 0 and abs_num_notes > 0: # going down the keys -> going left in the piano.
        octave -= 1
        note = note_cycle[-abs_num_notes]
    else: # going up the keys -> going right in the piano.
        note = note_cycle[abs_num_notes]
    note += str(octave)
    return note


def convert_midi_to_wav(folder, filename):
    '''
    convert midi file to wav file.
    browser can't play midi files so we need to convert to a format the browsers support.
    @param folder: (str) path to input folder.
    @param filename: (str) midi file name.
    @return: None.
    '''
    from midi2audio import FluidSynth

    fs = FluidSynth()
    fs.midi_to_audio(f'{folder}/{filename}.midi', f'{folder}/{filename}.wav')


