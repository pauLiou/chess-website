#database models, one for users and one for info
from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id')) #lowercase is for sql syntax

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    notes = db.relationship('Note')
    files = db.relationship('File')
    board = db.relationship('Board')

class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(1024*1024*10))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id')) #lowercase is for sql syntax
    date = db.Column(db.DateTime(timezone=True), default=func.now())

class Board(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.String(1024*1024*10))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id')) #lowercase is for sql syntax
    date = db.Column(db.DateTime(timezone=True), default=func.now())
