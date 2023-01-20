#storing the standard routes for the website for navigation
from flask import Blueprint, render_template, request, flash, Response, jsonify
from flask_login import login_required, current_user
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from .models import Note, File, Board
from . import db
import json
from .utils import *


class upload_file_form(FlaskForm):
    file = FileField('File')
    submit = SubmitField('Upload File')

#define that this script is a blueprint - separate the app out so we don't have to have views all defined in one file
views = Blueprint('views', __name__)

#defining our views
@views.route('/', methods=['GET','POST']) # decorator - the url endpoint route for this page
@login_required
def home():
    form = upload_file_form()
    if form.validate_on_submit():
        image_data_url = binary_to_image_data(form.file.data.mimetype,request.files[form.file.name].read())
        new_file = File(data=image_data_url, user_id=current_user.id)
        db.session.add(new_file)
        db.session.commit()
        flash('File has been uploaded!',category='success')
        predicted_img = run_chess_model(image_data_url)
        print(type(predicted_img))
        board_img = Board(data=predicted_img, user_id=current_user.id)
        db.session.add(board_img)
        db.session.commit()
        return render_template("home.html", form=form, user=current_user)
    # if request.method == 'POST':
    #     note = request.form.get('note')
    #     if len(note) < 1:
    #         flash('Note is too short!',category='error')
    #     else:
    #         new_note = Note(data=note, user_id=current_user.id)
    #         db.session.add(new_note)
    #         db.session.commit()
    #         flash('Note added',category='success')
    return render_template("home.html", form=form, user=current_user)

@views.route('/delete-image', methods=['POST'])
def delete_image():
    image_id = json.loads(request.data)['imageId']
    image = Board.query.get(image_id)
    if image and image.user_id == current_user.id:
        db.session.delete(image)
        db.session.commit()
        return jsonify({})
    return Response("Couldn't find image",status=400)

@views.route('/delete-file', methods=['POST'])
def delete_file():
    file_id = json.loads(request.data)['fileId']
    file = File.query.get(file_id)
    if file and file.user_id == current_user.id:
        db.session.delete(file)
        db.session.commit()
        return jsonify({})
    return Response("Couldn't find file",status=400)

