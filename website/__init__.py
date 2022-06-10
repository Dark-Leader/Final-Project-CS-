from flask import Flask
import os

TEMPLATE_FOLDER = "templates/"
UPLOAD_FOLDER = "data"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def create_app():
    app = Flask(__name__, template_folder=TEMPLATE_FOLDER)

    from .views import views
    app.register_blueprint(views, url_prefix='/')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.secret_key = os.urandom(12)

    return app
