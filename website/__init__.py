from flask import Flask
import os
import json
import torch

from algorithm.helper_functions import load_resnet101_model
from algorithm.Coordinator import Coordinator

TEMPLATE_FOLDER = "templates"
UPLOAD_FOLDER = "data"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
OUTPUT_FOLDER = "website\\static\\output"


def create_app():
    app = Flask(__name__, template_folder=TEMPLATE_FOLDER)

    from .views import views
    app.register_blueprint(views, url_prefix='/')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
    app.secret_key = os.urandom(12)
    return app


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


coordinator = create_coordinator()