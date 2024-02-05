from flask import request, url_for, redirect, render_template, jsonify, current_app
import matplotlib.pyplot as plt

from app.blueprints import BlueprintSingleton
from database.schemas import digit_images_schema, digit_images_many_schema, DigitImages
from utils import DateUtil


class NeuralNetworkBp(BlueprintSingleton):
    """ Implementation of neural network model related methods. """

    def digit_images(self, digit_image_id: int = None):
        if not digit_image_id:
            d_images = DigitImages.query.all()
            return jsonify(digit_images=digit_images_many_schema.dump(d_images))
        d_image = DigitImages.query.filter_by(id=digit_image_id).first()
        if d_image:
            return jsonify(digit_images=digit_images_schema.dump(d_image))
        else:
            return jsonify(message='There are no digit images with that id'), 404

    def upload_digit_image(self):
        image_file = request.files.get('file')
        image_file.save(current_app.config.get('UPLOAD_DIRECTORY') + 'digit.jpg')
        return redirect(url_for('nn.result'))

    def result(self):
        result = {'class': 1, 'probs': None, 'image': None}
        return render_template('nn/result.html', result=result)

    # gui views
    def predict(self):
        return render_template('nn/predict.html')

    def digit_images_table(self):
        return render_template("nn/digit_images_table.html")
