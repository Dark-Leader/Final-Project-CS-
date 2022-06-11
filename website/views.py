import json

from flask import Blueprint, render_template, send_from_directory, request, redirect, flash, current_app as app, \
    session, url_for
from werkzeug.utils import secure_filename
import os

from website import ALLOWED_EXTENSIONS

views = Blueprint("views", __name__)


@views.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(views.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@views.route("/")
def home():
    return render_template("home.html")


@views.route("/upload_file", methods=["GET", "POST"])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            # flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #
            # the coordinator did some work on the image
            #
            messages = json.dumps({'status': 'True'})
            session['messages'] = messages
            return redirect("/piano")

    return redirect("/")


def allowed_file(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@views.route("/piano", methods=["GET", "POST"])
def piano():
    print("inside piano")
    print(request.method)
    if request.method == "GET":
        if 'messages' not in session:
            return redirect("/")
        messages = json.loads(session['messages'])
        if 'status' not in messages or not messages['status']:
            return redirect("/")
        return render_template("piano.html")

