
from flask import Blueprint, render_template, send_from_directory, request, redirect, current_app as app, \
    session, abort
from werkzeug.utils import secure_filename
import os

from website import ALLOWED_EXTENSIONS, coordinator, NOTE_TO_BEATS, NOTE_TO_PITCH

KEYS = NOTE_TO_PITCH.keys()

views = Blueprint("views", __name__)


def allowed_file(file_name):
    return '.' in file_name and \
           file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
            file_name, extension = os.path.splitext(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #
            #
            try:
                notes = coordinator.process_image(filename, app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], NOTE_TO_BEATS)
            except Exception as e:
                print(e.message)
                return redirect("/")
            #
            # the coordinator did some work on the image
            #
            session['filename'] = file_name
            session['json_notes'] = [note.to_json() for note in notes]
            return redirect("/piano")

    return redirect("/")


@views.route("/piano", methods=["GET", "POST"])
def piano():
    if request.method == "GET":
        if 'filename' not in session or 'json_notes' not in session:
            return redirect("/")
        filename, json_notes = session['filename'], session['json_notes']
        print(filename, json_notes)

        return render_template("piano.html", file_name=filename, json_notes=json_notes, keys=KEYS)


@views.route("download_audio/<path:file_name>", methods=["GET"])
def download_audio(file_name):
    if not file_name:
        return abort(404)
    file_name = secure_filename(file_name)
    full_file_name = file_name + ".wav"
    print(full_file_name)
    base = app.root_path
    middle = app.config['OUTPUT_FOLDER']
    full_path = base + middle[middle.find('/'):]
    if not os.path.isfile(full_path + "/" + full_file_name):
        return abort(404)
    return send_from_directory(full_path, full_file_name, as_attachment=True)


@views.route("download_image/<path:file_name>", methods=["GET"])
def download_image(file_name):
    if not file_name:
        return abort(404)
    file_name = secure_filename(file_name)
    full_file_name = file_name + "_predictions.png"
    base = app.root_path
    middle = app.config['OUTPUT_FOLDER']
    full_path = base + middle[middle.find('/'):]
    if not os.path.isfile(full_path + "/" + full_file_name):
        return abort(404)
    return send_from_directory(full_path, full_file_name, as_attachment=True)

