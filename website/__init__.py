from flask import Flask
import os
import json
import torch

from algorithm.helper_functions import load_resnet101_model
from algorithm.Coordinator import Coordinator

TEMPLATE_FOLDER = "templates"
UPLOAD_FOLDER = "data"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
OUTPUT_FOLDER = "website/static/output"

with open("algorithm/utils/classes.json", "r") as fp:
    CLASSES = json.load(fp)

with open("algorithm/utils/note_to_pitch.json") as fp2:
    NOTE_TO_PITCH = json.load(fp2)

with open("algorithm/utils/beats_to_note.json", "r") as fp3:
    BEATS_TO_NOTE = json.load(fp3)

with open("algorithm/utils/note_to_beats.json", "r") as fp4:
    NOTE_TO_BEATS = json.load(fp4)


def create_app():
    app = Flask(__name__, template_folder=TEMPLATE_FOLDER)

    from .views import views
    app.register_blueprint(views, url_prefix='/')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
    app.secret_key = os.urandom(12)
    return app


def create_coordinator():
    model = load_resnet101_model("algorithm/ML/model.pt", len(CLASSES))
    CLASSES["invalid"] = -1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    coord = Coordinator(CLASSES, model, device, NOTE_TO_PITCH)
    return coord


coordinator = create_coordinator()