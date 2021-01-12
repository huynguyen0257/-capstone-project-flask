
from flask import Flask, send_file, request, jsonify, redirect, Blueprint
import cv2
import numpy as np
import base64
import time
import io
import sys
import argparse
import os
from pathlib import Path
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
sys.path.append('services/')
import mask_detection
mask_api = Blueprint('mask_api', __name__)


# load our serialized face detector model from disk
prototxtPath = str(Path().absolute()) + '/services/models/deploy.prototxt'
weightsPath = str(Path().absolute()) + \
    '/services/models/res10_300x300_ssd_iter_140000.caffemodel'
maskModelPath = str(Path().absolute()) + '/services/models/mask_detector.model'

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(maskModelPath)

# Detect face with return is x,y,w,h of the face location


@mask_api.route('/detectfacemask', methods=['POST'])
def returnDetectFaceMask():
    start = time.time()
    data = request.get_json()
    if data is None:
        print("No valid request body, json missing!")
        return jsonify({'error': 'No valid request body, json missing!'})
    else:
        img_data = data['image']
        decoded_data = base64.b64decode(img_data)
        np_data = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mask_detection.detect_and_predict_mask(img, faceNet, maskNet)
        print('time face: ', time.time() - start)
        return jsonify({'info':result, 'people':['unknown']})
