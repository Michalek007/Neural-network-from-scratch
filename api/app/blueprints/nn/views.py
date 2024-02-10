import flask_login

from app.blueprints.nn import nn as nn
from app.blueprints.nn.nn_bp import NeuralNetworkBp


@nn.route('/digit_images/<int:digit_image_id>/', methods=['GET'])
@nn.route('/digit_images/', methods=['GET'])
def digit_images(digit_image_id: int = None):
    """ Returns digit image with given id or if not specified list of all digit images from database.
        Input args: /digit_id/.
        Output keys: digit_images {id, path, digit}
    """
    return NeuralNetworkBp().digit_images(digit_image_id=digit_image_id)


@nn.route('/upload_digit_image/', methods=['POST'])
def upload_digit_image():
    """ POST method.
        Allows to upload image file, which will be saved in upload directory.
        Redirects to /result/ url.
    """
    return NeuralNetworkBp().upload_digit_image()


@nn.route('/result/', methods=['GET'])
def result():
    """ Predicts digit based on uploaded image.
        Returns result.html template.
    """
    return NeuralNetworkBp().result()


@nn.route('/digit_images_table/', methods=['GET'])
def digit_images_table():
    return NeuralNetworkBp().digit_images_table()


@nn.route('/predict/', methods=['GET'])
def predict():
    return NeuralNetworkBp().predict()
