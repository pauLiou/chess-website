from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login.login_manager import LoginManager
from werkzeug.utils import secure_filename
import os

basedir = os.path.abspath(os.path.dirname(__file__))
db = SQLAlchemy() # create a database
DB_NAME = "database.db" # name the database
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'} # allowed file type uploads

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'weorfhwPFUHWEOFIJOIUHOIUFhbejyUHGOIH'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///../website/{DB_NAME}'
    db.init_app(app)

    from .auth import auth
    from .views import views  # importing the blueprints
    
    app.register_blueprint(views,url_prefix='/') # registering the blueprints
    app.register_blueprint(auth,url_prefix='/')

    from .models import User, Note

    with app.app_context():
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app
