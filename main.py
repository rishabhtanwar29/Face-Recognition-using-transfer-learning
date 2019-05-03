import os
import io
import numpy as np
import sqlite3
from capture import take_image
from face_location import get_face_locations
from vgg_net import preprocess_image, loadVggFaceModel, findEuclideanDistance


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)
vgg_face_descriptor = loadVggFaceModel()
epsilon = 50.0

print('Do you want to verify? (y/n)')
condition = True if input() == 'y' else False

while condition:
    print('Press \'q\' to take picture')
    frame = take_image()
    get_face_locations(frame)
    cur_path = os.path.abspath(os.path.dirname(__file__))
    cur_path = os.path.join(cur_path, "face_detected")
    faces = os.listdir(cur_path)
    representations = []
    for face in faces:
        img_representation = vgg_face_descriptor.predict(preprocess_image(os.path.join(cur_path, face)))[0, :]
        representations.append(img_representation)
        os.unlink(os.path.join(cur_path,face))
    conn = sqlite3.connect('Database/database.db', detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.execute("SELECT * FROM Students")
    for img in representations:
        for row in cursor:
            average_euclidean_distance = 0.0
            for i in range(2,12):
                img_rep = row[i]
                average_euclidean_distance += float(findEuclideanDistance(img, img_rep))
            average_euclidean_distance /= 10.0
            if average_euclidean_distance < epsilon:
                print("Hello "+row[1]+" !")
                print(average_euclidean_distance)
    print("Do you want to continue? (y/n)")
    condition = True if input() == 'y' else False