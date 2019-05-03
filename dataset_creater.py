import os
import numpy as np
import sqlite3
import io
from capture import take_image
from face_location import get_face_locations
from vgg_net import loadVggFaceModel, preprocess_image


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
"""conn.execute('''CREATE TABLE Students
         (ID INT PRIMARY KEY     NOT NULL,
         NAME           TEXT    NOT NULL,
         VECTOR_0       array,
         VECTOR_1       array,
         VECTOR_2       array,
         VECTOR_3       array,
         VECTOR_4       array,
         VECTOR_5       array,
         VECTOR_6       array,
         VECTOR_7       array,
         VECTOR_8       array,
         VECTOR_9       array);''')"""


def insertOrUpdate(name,rep_arr):
    conn = sqlite3.connect('Database/database.db', detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.execute("SELECT * FROM Students WHERE NAME = ?",(name,))
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if isRecordExist == 1:
        for i in range(len(rep_arr)):
            conn.execute("UPDATE Students SET VECTOR_"+str(i)+" = ? WHERE NAME = ? ", (rep_arr[i], name,))
    else:
        cursor = conn.execute('SELECT MAX(ID) FROM Students')
        id = 0
        for row in cursor:
            id = row[0]+1
        conn.execute("INSERT INTO Students (ID,NAME,VECTOR_0) values (?,?,?)", (id, name, rep_arr[0],))
        for i in range(1, len(rep_arr)):
            conn.execute("UPDATE Students SET VECTOR_" + str(i) + " = ? WHERE NAME = ? ", (rep_arr[i], name,))
    conn.commit()
    conn.close()


vgg_face_descriptor = loadVggFaceModel()
name = input("Enter your name ")
sampleNum = 0
rep_arr = []

while sampleNum<10:
    print('Press \'q\' to take picture')
    frame = take_image()
    get_face_locations(frame)
    cur_path = os.path.abspath(os.path.dirname(__file__))
    cur_path = os.path.join(cur_path, "face_detected")
    faces = os.listdir(cur_path)
    img_representation = vgg_face_descriptor.predict(preprocess_image(os.path.join(cur_path, faces[0])))[0, :]
    os.unlink(os.path.join(cur_path, faces[0]))
    rep_arr.append(img_representation)
    sampleNum += 1
    if sampleNum == 10:
        insertOrUpdate(name, rep_arr)