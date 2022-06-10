import json
import torch
import os
from algorithm.Coordinator import Coordinator
from algorithm.helper_functions import load_resnet101_model

from website import create_app


def create_coordinator():

    with open("algorithm/utils/classes.json", "r") as fp:
        classes = json.load(fp)
    model = load_resnet101_model("algorithm/ML/model.pt", len(classes))
    classes["invalid"] = -1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    with open("algorithm/utils/note_to_pitch.json") as f:
        note_to_pitch = json.load(f)
    coord = Coordinator(classes, model, device, note_to_pitch)
    return coord


app = create_app()

#coordinator = create_coordinator()



if __name__ == '__main__':
    #c = create_coordinator()
    app.run(debug=True)