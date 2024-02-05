from datetime import datetime
import os

from cli.groups import BaseGroup
from lib_objects import bcrypt
from database.schemas import User, Performance, DigitImages


class SeedGroup(BaseGroup):
    """ Implements methods responsible for seeding database. """
    bcrypt = bcrypt

    def init(self):
        admin = User(username='admin', pw_hash=self.bcrypt.generate_password_hash('admin'))

        self.db.session.add(admin)
        self.db.session.commit()
        print('Database seeded!')

    def users(self):
        user1 = User(username='test1', pw_hash=bcrypt.generate_password_hash('test'))
        user2 = User(username='test2', pw_hash=bcrypt.generate_password_hash('test'))
        user3 = User(username='test3', pw_hash=bcrypt.generate_password_hash('test'))

        self.db.session.add(user1)
        self.db.session.add(user2)
        self.db.session.add(user3)
        self.db.session.commit()
        print('Database seeded!')

    def params(self):
        param1 = Performance(timestamp=datetime.now(), memory_usage=50, cpu_usage=20, disk_usage=50)
        param2 = Performance(timestamp=datetime.now(), memory_usage=50, cpu_usage=20, disk_usage=50)
        param3 = Performance(timestamp=datetime.now(), memory_usage=50, cpu_usage=20, disk_usage=50)

        self.db.session.add(param1)
        self.db.session.add(param2)
        self.db.session.add(param3)
        self.db.session.commit()
        print('Database seeded!')

    def digit_images(self):
        image1 = DigitImages(path='C:\\Users\\Public\\Projects\\MachineLearning\\NeuralNetwork\\archive\\trainingSample\\img_0.jpg', digit=0)
        image2 = DigitImages(path='C:\\Users\\Public\\Projects\\MachineLearning\\NeuralNetwork\\archive\\trainingSample\\img_1.jpg', digit=1)
        image3 = DigitImages(path='C:\\Users\\Public\\Projects\\MachineLearning\\NeuralNetwork\\archive\\trainingSample\\img_2.jpg', digit=0)

        self.db.session.add(image1)
        self.db.session.add(image2)
        self.db.session.add(image3)
        self.db.session.commit()
        print('Database seeded!')
