import sys
sys.path.append('services/')
import image_process
import face_detection
from pathlib import Path
import face_recognition
from PIL import Image, ImageDraw
import dlib
import io
import time
import base64
import numpy as np
import cv2
from flask import Flask, send_file, request, jsonify, redirect, Blueprint
import requests



face_api = Blueprint('face_api', __name__)

# array of disable user
arr_disable_name=[]
arr_known_encode_face_global = face_detection.encode_known_face_from_firebase(arr_disable_name)
# Remove duplicate
arr_disable_name = list(dict.fromkeys(arr_disable_name))


# training face detection by building ---------------------------------------------------------------------------
known_face_buildings = {}
# GET BUILDINGS
x_rq = requests.get('http://localhost:8888/api/building')
buildings = x_rq.json()

for building in buildings:
    known_face_buildings[building['Id']] = face_detection.encode_known_face_from_firebase_by_building(building['Id'])

face_detection.encode_security_guard_known_face_from_firebase_to_all_building(
    arr_disable_name,
    buildings,
    known_face_buildings
)
# ----------------------------------------------------------------------------------------------------------------

predictor_face_68_model_path = str(
    Path().absolute()) + "/services/models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_face_68_model_path)
detector = dlib.get_frontal_face_detector()

list_username_enhance = []
# Preprocess headpose
arr_preprecess_headpose = face_detection.preprocessHeadPoseV2()


@face_api.route('/OnSuccessFaceCreated', methods=['POST'])
def onSuccessFaceCreated():
    data = request.get_json()

    username = data['Username']
    for user_loaded in list_username_enhance:
        if(username == user_loaded[0]):
            return 'user is loaded'
    isRelative = data['IsRelative']
    buildingId = data['BuildingId']
    arr = data['ImageUrls']
    arr_user_img = []
    if(len(arr) > 0):
        arr_user_img = face_detection.load_images_from_firebase(arr, isRelative)
        print('================================list url user img: ', len(arr_user_img),'================================')
    print('OnSuccessFaceCreated user:', username)
    # get encode known face all link in firebase
    i = 0
    while i < len(arr_disable_name):
        if(arr_disable_name[i] in username):
            print('user active again in all:', username)
            face_detection.disable_know_face_by_username(username, arr_known_encode_face_global, True)
            if (buildingId != None):
                print('student',username,' active again in building', buildingId)
                face_detection.encode_new_face(known_face_buildings[buildingId], arr_user_img, isRelative)
            if (buildingId == None):
                print('security',username,' active again in all building')
                for building in buildings:
                    face_detection.disable_know_face_by_username(
                    username, known_face_buildings[building['Id']], True)
            # REMOVE RA KHOI arr_disable_name
            del arr_disable_name[i]
            print('active without re-train')
            return 'create success'
        else:
            i = i+ 1
      
    print('encode_new_face in global')
    face_detection.encode_new_face(
        arr_known_encode_face_global, arr_user_img, isRelative)

    # This is student & relative
    if (buildingId != None):
        print('encode_new_face in 1 building :',buildingId)
        face_detection.encode_new_face(
        known_face_buildings[buildingId], arr_user_img, isRelative)

    # This is security man
    if (buildingId == None and isRelative == False):
        print('encode_new_face in all building')
        for building in buildings:
            face_detection.encode_new_face(
            known_face_buildings[building['Id']], arr_user_img, isRelative)
    # print('length:' + len(result))
    return 'create success'

@face_api.route('/getInfo', methods=['GET'])
def getInfo():
    result = []
    result.append({'disable_name':arr_disable_name})
    result.append({'Global': {'Username': arr_known_encode_face_global['faceName'], 'isActive': arr_known_encode_face_global['faceisActive']}})
    for building in buildings:
        # if(arr_disable_name[i]+'@' in username):
        result.append({'building '+ str(building['Id']): {'Username': known_face_buildings[building['Id']]['faceName'], 'isActive': known_face_buildings[building['Id']]['faceisActive']}})
    
    return jsonify(result)


@face_api.route('/OnCheckoutSuccess', methods=['POST'])
def OnCheckoutSuccess():
    data = request.get_json()

    username = data['Username']
    buildingId = data['BuildingId']
    try:
        isRemove = data['IsRemove']
    except:
        isRemove = None

    if (isRemove == None):
        arr_disable_name.append(username.split('@')[0])
    print('remove user', username)
    
    # Disable student & security man from system
    face_detection.disable_know_face_by_username(
        username, arr_known_encode_face_global, False)
    
    # Remove relative out system
    if (isRemove != None):
        print('remove relative from global|', username)
        face_detection.remove_know_face_by_username(username, arr_known_encode_face_global)
    
    # Remove student & relative from building
    if (buildingId != None):
        print('remove student + relative from building|', username)
        face_detection.remove_know_face_by_username(username, known_face_buildings[buildingId])
    # Disable security man from all building
    if (buildingId == None):
        print('Disable security man from all building|', username)
        for building in buildings:
            face_detection.disable_know_face_by_username(
            username, known_face_buildings[building['Id']], False)
            
    return 'checkout success'

@face_api.route('/OnRemoveFaceImage', methods=['POST'])
def retrain_face():
    data = request.get_json()
    username = data['Username']
    buildingId = data['BuildingId']
    
    if (buildingId != None):
        print('remove student + relative global', username)
        face_detection.remove_know_face_by_username(username, arr_known_encode_face_global)
        print('remove student + relative from building|', username)
        face_detection.remove_know_face_by_username(username, known_face_buildings[buildingId])

    if (buildingId == None):
        print('remove guard global', username)
        face_detection.remove_know_face_by_username(username, arr_known_encode_face_global)
        print('remove guard from building', username)
        for building in buildings:
            face_detection.remove_know_face_by_username(username, known_face_buildings[building['Id']])

    return 'OnRemoveFaceImage success'

# ----------- Face detection API ----------

# Detect Face with base64

# Detect face with return is x,y,w,h of the face location
@face_api.route('/detectfacelocation', methods=['POST'])
def returnDetectFaceLocation():
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

        result = face_detection.detect_face_location(
            img, arr_known_encode_face_global)

        print('time face: ', time.time() - start)
        return jsonify(result)

# Detect face with return is x,y,w,h of the face location

# retrain face

@face_api.route('/detectfacelocationbybuilding', methods=['POST'])
def returnDetectFaceLocationByBuilding():
    start = time.time()
    data = request.get_json()
    if data is None:
        print("No valid request body, json missing!")
        return jsonify({'error': 'No valid request body, json missing!'})
    else:
        img_data = data['image']
        buildingId = data['BuildingId']
        decoded_data = base64.b64decode(img_data)
        np_data = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

        result = face_detection.detect_face_location(
            img, known_face_buildings[buildingId])
        print("buildingId:",buildingId)
        print('time face: ', time.time() - start)
        return jsonify(result)


# ----------- Roll Pitch Yaw detection API ----------


# Return roll pitch yaw of face version 2


@face_api.route('/faceregisterV2', methods=['POST'])
def returnRollPitchYawV2():

    # get base64 image from request body
    data = request.get_json()
    img_data = data['image']

    # base64 to cv2 image
    decoded_data = base64.b64decode(img_data)
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

    result = face_detection.processHeadPoseV2(img, predictor, detector,
                                              arr_preprecess_headpose)

    return result
