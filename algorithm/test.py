import cv2
from math import pi, sqrt
import os
import json
from midiutil import MIDIFile
from algorithm.helper_functions import round_to_closest_half
import math


def get_note_locations2(image, spacing, is_whole=False):

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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
    cv2.imwrite("lines1.png", lines1)
    cv2.imwrite("lines2.png", lines2)
    #cv2.waitKey(0)

    for cntr in cntrs:
        area = cv2.contourArea(cntr)
        if area > spacing and area < (spacing * 2.2) ** 2:
            # get centroid
            M = cv2.moments(cntr)
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            pt = "(" + str(cx) + "," + str(cy) + ")"
            # fit ellipse

            ellipse = cv2.fitEllipse(cntr)
            (x, y), (minor_axis, major_axis), angle = ellipse
            poly = cv2.ellipse2Poly((int(x), int(y)), (int(major_axis / 2), int(minor_axis / 2)), int(angle), 0, 360, 1)
            similarity = cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), cntr, cv2.CONTOURS_MATCH_I1,0)
            print(similarity, area)
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
    num_notes = math.floor(num_jumps * 2)
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



def main():
    beats_to_note = {4: "whole",
                     2: "half",
                     1: "quarter",
                     0.5: "eighth",
                     0.25: "sixteenth"}

    with open("utils/beats_to_note.json", "w") as fp:
        json.dump(beats_to_note, fp)

    note_to_beats = {"whole": 4,
                     "half": 2,
                     "quarter": 1,
                     "eighth": 0.5,
                     "sixteenth": 0.25,
                     "rest_whole": 4,
                     "rest_half": 2,
                     "rest_quarter": 1,
                     "rest_eighth": 0.5,
                     "rest_sixteenth": 0.25}

    with open("utils/note_to_beats.json", "w") as fp2:
        json.dump(note_to_beats, fp2)




if __name__ == '__main__':
    main()