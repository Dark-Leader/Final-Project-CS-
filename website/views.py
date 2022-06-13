import json

from flask import Blueprint, render_template, send_from_directory, request, redirect, flash, current_app as app, \
    session, url_for, abort
from werkzeug.utils import secure_filename
import os

from website import ALLOWED_EXTENSIONS, coordinator

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
            #
            success = True
            try:
                coordinator.process_image(filename, app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'])
            except Exception as e:
                success = False
            #
            # the coordinator did some work on the image
            #
            if not success:
                return redirect("/")
            messages = json.dumps({'status': 'True', 'filename': filename})
            session['messages'] = messages
            return redirect("/piano")

    return redirect("/")


def allowed_file(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@views.route("/piano", methods=["GET", "POST"])
def piano():
    if request.method == "GET":
        if 'messages' not in session:
            return redirect("/")
        messages = json.loads(session['messages'])
        if 'status' not in messages or not messages['status'] or 'filename' not in messages:
            return redirect("/")
        filename = messages['filename']
        print(filename)
        file_name_no_ext = filename.split('.')[0]
        full_name = f"{file_name_no_ext}_predictions"
        print(full_name)
        return render_template("piano.html", file_name=full_name)


@views.route("download_audio/<path:file_name>", methods=["GET"])
def download_audio(file_name):
    if not file_name:
        return abort(404)
    full_file_name = file_name[0:file_name.find('_predictions')] + ".midi"
    print(file_name)
    base = app.root_path
    middle = app.config['OUTPUT_FOLDER']
    full_path = base + middle[middle.find('\\'):]
    if not os.path.isfile(full_path + "\\" + full_file_name):
        return abort(404)
    return send_from_directory(full_path, full_file_name, as_attachment=True)


@views.route("download_image/<path:file_name>", methods=["GET"])
def download_image(file_name):
    if not file_name:
        return abort(404)
    full_file_name = file_name + ".png"
    base = app.root_path
    middle = app.config['OUTPUT_FOLDER']
    full_path = base + middle[middle.find('\\'):]
    if not os.path.isfile(full_path + "\\" + full_file_name):
        return abort(404)
    return send_from_directory(full_path, full_file_name, as_attachment=True)

