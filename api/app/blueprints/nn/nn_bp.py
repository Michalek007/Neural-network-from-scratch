from io import BytesIO

from flask import request, url_for, redirect, render_template, jsonify, current_app
import matplotlib.pyplot as plt
from PIL import UnidentifiedImageError, Image
import base64
import numpy as np

from app.blueprints import BlueprintSingleton
from database.schemas import digit_images_schema, digit_images_many_schema, DigitImages
from neural_network import NeuralNetwork


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
        network = NeuralNetwork(current_app.config.get('BASEDIR') + '\\neural_network\\models\\mnist_model.json')

        try:
            x_test = plt.imread(current_app.config.get('UPLOAD_DIRECTORY') + 'digit.jpg')
        except UnidentifiedImageError:
            return jsonify(message='Incorrect file format. Expected: .jpg'), 408

        x_test = x_test.reshape(28 * 28)
        x_test = x_test.astype('float32')
        x_test /= 255
        x_test = np.array([x_test])

        out = network.predict(x_test)

        predicted_class = np.argmax(out)
        probability = np.max(out)

        # with open(current_app.config.get('UPLOAD_DIRECTORY') + 'digit.jpg', 'rb') as f:
        #     image_data = base64.b64encode(f.read())

        result = {'class': predicted_class, 'probs': probability, 'image': None}
        return render_template('nn/result.html', result=result)

    # gui views
    def predict(self):
        return render_template('nn/predict.html')

    def digit_images_table(self):
        return render_template('nn/digit_images_table.html')
