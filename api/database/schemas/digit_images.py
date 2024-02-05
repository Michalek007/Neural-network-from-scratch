from sqlalchemy import Column, Integer, String, Float
from lib_objects import ma
from database import db


class DigitImages(db.Model):
    """ Table for digit images.
        Fields -> 'id', 'path'
    """
    __tablename__ = 'digit_images'
    id = Column(Integer, primary_key=True)
    path = Column(String)
    digit = Column(Integer)


class DigitImagesSchema(ma.Schema):
    class Meta:
        fields = ('id', 'path', 'digit')


digit_images_schema = DigitImagesSchema()
digit_images_many_schema = DigitImagesSchema(many=True)
