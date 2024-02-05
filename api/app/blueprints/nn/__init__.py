from flask import Blueprint

nn = Blueprint('nn',
               __name__,
               # url_prefix='/nn',
               template_folder='templates')

from app.blueprints.nn import views
