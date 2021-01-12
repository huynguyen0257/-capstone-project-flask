import urllib.request as ur
import face_recognition
import numpy as np
from PIL import Image
import requests
from io import BytesIO

response = requests.get('https://storage.googleapis.com/sdms-captone-4ab5b.appspot.com/users-face-images/huy1234@gmail.com/2020-11-12T05:14:25.793Z.jpg')
img = np.asarray(Image.open(BytesIO(response.content)))
img = img[:, :, ::-1]

print(len(face_recognition.face_encodings(img)))
# user_face_encoding = face_recognition.face_encodings(user_image)[0]