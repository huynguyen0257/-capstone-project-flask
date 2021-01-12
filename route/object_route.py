from flask import Flask, send_file, request, jsonify, redirect, Blueprint
import cv2
import numpy as np
import base64
import time
import io
import sys
from PIL import Image, ImageDraw

sys.path.append('services/')
import object_detection
import image_process

object_api = Blueprint('object_api', __name__)


# ----------Object detection API----------


# *********Postman***********
@object_api.route('/objectdetectiontest', methods=['POST'])
def returnObjectDetectionImage():
    img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8),
                       cv2.IMREAD_UNCHANGED)

    result = object_detection.detect_object_custom(img)

    file_object = image_process.convertfromnptoimage(result)

    return send_file(file_object, mimetype='image/PNG')

@object_api.route('/objectdetectiontestfull', methods=['POST'])
def returnObjectDetectionImageFull():
    img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8),
                       cv2.IMREAD_UNCHANGED)

    result = object_detection.detect_object_full(img)

    file_object = image_process.convertfromnptoimage(result)

    return send_file(file_object, mimetype='image/PNG')

@object_api.route('/objectdetectiontesttiny', methods=['POST'])
def returnObjectDetectionImageTiny():
    img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8),
                       cv2.IMREAD_UNCHANGED)

    result = object_detection.detect_object_tiny(img)

    file_object = image_process.convertfromnptoimage(result)

    return send_file(file_object, mimetype='image/PNG')

# ************Base64***********
@object_api.route('/detectobjectbase64', methods=['POST'])
def returnObjectDetectBase64():
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

        result = object_detection.detect_object_custom(img)

        im = Image.fromarray(result.astype('uint8'))
        rawBytes = io.BytesIO()
        im.save(rawBytes, "PNG")
        rawBytes.seek(0)
        im_base64 = base64.b64encode(rawBytes.read()).decode('utf-8')
        result = {"image": im_base64, "people": "dao"}
        print(time.time() - start)
        return result

@object_api.route('/detectobjectbase64full', methods=['POST'])
def returnObjectDetectBase64Full():
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

        result = object_detection.detect_object_full(img)

        im = Image.fromarray(result.astype('uint8'))
        rawBytes = io.BytesIO()
        im.save(rawBytes, "PNG")
        rawBytes.seek(0)
        im_base64 = base64.b64encode(rawBytes.read()).decode('utf-8')
        result = {"image": im_base64, "people": "dao"}
        print(time.time() - start)
        return result

@object_api.route('/detectobjectbase64tiny', methods=['POST'])
def returnObjectDetectBase64Tiny():
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

        result = object_detection.detect_object_tiny(img)

        im = Image.fromarray(result.astype('uint8'))
        rawBytes = io.BytesIO()
        im.save(rawBytes, "PNG")
        rawBytes.seek(0)
        im_base64 = base64.b64encode(rawBytes.read()).decode('utf-8')
        result = {"image": im_base64, "people": "dao"}
        print(time.time() - start)
        return result

@object_api.route('/detectobjectbase64tinylocation', methods=['POST'])
def returnObjectDetectBase64TinyLocation():
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

        result = object_detection.detect_object_location(img)
        print(result)
        
        print("time object: ",time.time() - start)
        return jsonify(result)

@object_api.route('/detectprohibitedcustomlocation', methods=['POST'])
def returnProhibitedCustomLocation():
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

        result = object_detection.detect_prohibited_location(img)

        
        print("time object: ",time.time() - start)
        return jsonify(result)

@object_api.route('/detectprohibitedfulllocation', methods=['POST'])
def returnProhibitedFullLocation():
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

        result = object_detection.detect_prohibited_location_full(img)

        
        print("time object: ",time.time() - start)
        return jsonify(result)

