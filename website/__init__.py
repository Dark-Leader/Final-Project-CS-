from flask import Flask
import os
import torch

from algorithm.helper_functions import load_resnet101_model
from algorithm.Coordinator import Coordinator
from config import settings

TEMPLATE_FOLDER = "templates"
UPLOAD_FOLDER = "data"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
OUTPUT_FOLDER = "website/static/output"

CLASSES = settings['classes']
NOTE_TO_PITCH = settings['note_to_pitch']
BEATS_TO_NOTE = settings['beats_to_note']
NOTE_TO_BEATS = settings['note_to_beats']


def create_app():
    app = Flask(__name__, template_folder=TEMPLATE_FOLDER)

    from .views import views
    app.register_blueprint(views, url_prefix='/')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
    app.secret_key = os.urandom(12)
    return app


def create_coordinator():
    model = load_resnet101_model(settings['model_path'], len(CLASSES))
    CLASSES["invalid"] = -1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    coord = Coordinator(CLASSES, model, device, NOTE_TO_PITCH)
    return coord


coordinator = create_coordinator()