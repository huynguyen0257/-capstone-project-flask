
import face_recognition
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential, load_model
import _pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
import dlib
import numpy as np
from PIL import Image, ImageDraw
import urllib.request as ur
import math
import cv2
import sys
import os

sys.path.append('faces/')

face_pose_pkl_path = str(Path().absolute()) + '/services/models/samples.pkl'
face_pose_model_path = str(Path().absolute()) + '/services/models/model.h5'

tolerance = 0.38


class Person:
    def __init__(self, name, encoding_face, isRelative, isActive):
        self.name = name
        self.encoding_face = encoding_face
        self.isRelative = isRelative
        self.isActive = isActive

class MyImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.img_name = img_name

    def get_file_name(self):
        return self.img_name.split('/')[-1].split('.')[0]

    def get_name(self):
        filename = self.img_name.split('/')[-1].split('.')[0]
        return filename.split('-')[0].split('@')[0]


class MyImageFireBase:
    def __init__(self, img_link, user_face_encoding):
        self.img_link = img_link
        self.user_face_encoding = user_face_encoding

    def get_name(self):
        image_id = self.img_link.replace(
            "https://storage.googleapis.com/sdms-captone-4ab5b.appspot.com/users-face-images/",
            "").split("/")[0].split('@')[0]
        return image_id


class MyImageFireBaseRelative:
    def __init__(self, img_link, user_face_encoding):
        self.img_link = img_link
        self.user_face_encoding = user_face_encoding

    def get_name(self):
        image_id = self.img_link.replace(
            "https://storage.googleapis.com/sdms-captone-4ab5b.appspot.com/users-face-images/",
            "").split("/")[1]
        return image_id

    def get_real_name(self):
        x_rq = requests.get(
            'http://localhost:8888/api/Relative?IdentityCardNumber='+self.get_name).json()
        real_name = x_rq['Name']
        return real_name

# return a list of MyImageFireBase


def load_images_from_firebase(arr_link_images, isRelative):
    image_object_list = []
    if(isRelative == False):
        for url in arr_link_images:
            try:
                response = ur.urlopen(url)
                user_image = face_recognition.load_image_file(response)
                user_face_encoding = face_recognition.face_encodings(user_image)[0]
                image_object = MyImageFireBase(url,user_face_encoding)
                image_object_list.append(image_object)
            except:
                print('Error link', url)
        return image_object_list
    else:
        for url in arr_link_images:
            try:
                response = ur.urlopen(url)
                user_image = face_recognition.load_image_file(response)
                user_face_encoding = face_recognition.face_encodings(user_image)[0]
                image_object = MyImageFireBaseRelative(url,user_face_encoding)
                image_object_list.append(image_object)
            except:
                print('Error link', url)
        return image_object_list

# return a global for start project

def encode_security_guard_known_face_from_firebase_to_all_building(arr_disable_name,buildings,known_face_buildings):
    try:

        x_api = ('http://localhost:8888/api/User/SecurityGuard/FaceImages?IsActive=true')
        z_api = ('http://localhost:8888/api/User/SecurityGuard/FaceImages?IsActive=false')
        # fetch list user face image link from firebase
        x = requests.get(x_api.replace(' ', ''))
        result_x = x.json()
        urls_x = result_x['ImageURLs']

        # fetch list disable user face image link from firebase
        z = requests.get(z_api.replace(' ', ''))
        result_z = z.json()
        urls_z = result_z['ImageURLs']
        
      

        arr_security_guard_img = []
        arr_disable_security_guard_img = []

        # load all image from url
        if(len(urls_x) > 0):
            arr_security_guard_img = load_images_from_firebase(urls_x, False)
            print('================================list url of AVAILABLE security guard img to all building: ', len(arr_security_guard_img),'================================')
        if(len(urls_z) > 0):
            arr_disable_security_guard_img = load_images_from_firebase(urls_z, False)
            print('================================list url of DISABLE security guard img to all building: ', len(arr_disable_security_guard_img),'================================')    
        
        # train all face in list
        count_security = 0
        count_disable_security = 0

        for image in arr_security_guard_img:
            try:
                # Load image from image path and encode to store face
                if (len(image.user_face_encoding) != 0):
                    user = Person(image.get_name(), image.user_face_encoding, False, True)
                    for building in buildings:
                        known_face_buildings[building['Id']]['personList'].append(user)
                        known_face_buildings[building['Id']]['faceEncoding'].append(
                            user.encoding_face)
                        known_face_buildings[building['Id']]['faceName'].append(user.name)
                        known_face_buildings[building['Id']]['faceRelative'].append(user.isRelative)
                        known_face_buildings[building['Id']]['faceisActive'].append(user.isActive)
                    count_security = count_security + 1
                    arr_disable_name.append(user.name)
            except:
                print('Error link: ', image.img_link)

        for image in arr_disable_security_guard_img:
            try:
                # Load image from image path and encode to store face
                if (len(image.user_face_encoding) != 0):
                    user = Person(image.get_name(), image.user_face_encoding, False, False)
                    for building in buildings:
                        known_face_buildings[building['Id']]['personList'].append(user)
                        known_face_buildings[building['Id']]['faceEncoding'].append(
                            user.encoding_face)
                        known_face_buildings[building['Id']]['faceName'].append(user.name)
                        known_face_buildings[building['Id']]['faceRelative'].append(user.isRelative)
                        known_face_buildings[building['Id']]['faceisActive'].append(user.isActive)
                    arr_disable_name.append(user.name)
                    count_disable_security = count_disable_security + 1

            except:
                print('Error link: ', image.img_link)
        
        # print('Success Load all', count_security, 'security to building',building['Id'])
        # print('Success Load all', count_disable_security, 'disable security to building',building['Id'])
    except:
        print('Fail Load all', count_security+count_disable_security, 'to all building list')


def encode_known_face_from_firebase(arr_disable_name):
    # fetch list user face image link from firebase

    list_known_person = {
        'personList': [],
        'faceEncoding': [],
        'faceName': [],
        'faceRelative': [],
        'faceisActive':[]
    }
    try:

        x_api = ('http://localhost:8888/api/User/FaceImages')
        y_api = ('http://localhost:8888/api/Relative/FaceImages')
        z_api = ('http://localhost:8888/api/DeactiveUser/FaceImages')
        # fetch list user face image link from firebase
        x = requests.get(x_api.replace(' ', ''))
        result_x = x.json()
        urls_x = result_x['ImageURLs']

        # fetch list relatives face image link from firebase
        y = requests.get(y_api.replace(' ', ''))
        result_y = y.json()
        urls_y = result_y['ImageURLs']

        # fetch list disable user face image link from firebase
        z = requests.get(z_api.replace(' ', ''))
        result_z = z.json()
        urls_z = result_z['ImageURLs']
        
        arr_user_img = []
        arr_disable_user_img = []
        arr_relative_img = []

        # load all image from url
        if(len(urls_x) > 0):
            arr_user_img = load_images_from_firebase(urls_x, False,)
            print('================================ user img in all: ', len(arr_user_img),'================================')
        if(len(urls_z) > 0):
            arr_disable_user_img = load_images_from_firebase(urls_z, False)
            print('================================list url relative img in all: ', len(arr_disable_user_img),'================================')    
        if(len(urls_y) > 0):
            arr_relative_img = load_images_from_firebase(urls_y, True)
            print('================================list url disable img in all: ', len(arr_relative_img),'================================')
        # train all face in list
        count_user = 0
        count_relative = 0
        count_disable = 0

        for image in arr_user_img:
            try:
                # Load image from image path and encode to store face

                if (len(image.user_face_encoding) != 0):
                    user = Person(image.get_name(), image.user_face_encoding, False, True)
                    list_known_person['personList'].append(user)
                    list_known_person['faceEncoding'].append(
                        user.encoding_face)
                    list_known_person['faceName'].append(user.name)
                    list_known_person['faceRelative'].append(user.isRelative)
                    list_known_person['faceisActive'].append(user.isActive)
        
                    print('load firebase user ', image.get_name())
                    count_user = count_user + 1
                    # arr_disable_name.append(user.name)
            except:
                print('Error user link: ', image.img_link)

        for image in arr_relative_img:
            try:
                # Load image from image path and encode to store face

                if (len(image.user_face_encoding) != 0):
                    user = Person(image.get_name(), image.user_face_encoding, True, True)
                    list_known_person['personList'].append(user)
                    list_known_person['faceEncoding'].append(
                        user.encoding_face)
                    list_known_person['faceName'].append(user.name)
                    list_known_person['faceRelative'].append(user.isRelative)
                    list_known_person['faceisActive'].append(user.isActive)
                    # arr_disable_name.append(user.name)
                    print('load firebase relative :', image.get_name())
                    count_relative = count_relative + 1

            except:
                print('Error relative link: ', image.img_link)

        for image in arr_disable_user_img:
            try:
                # Load image from image path and encode to store face

                if (len(image.user_face_encoding) != 0):
                    user = Person(image.get_name(), image.user_face_encoding, False, False)
                    list_known_person['personList'].append(user)
                    list_known_person['faceEncoding'].append(
                        user.encoding_face)
                    list_known_person['faceName'].append(user.name)
                    list_known_person['faceRelative'].append(user.isRelative)
                    list_known_person['faceisActive'].append(user.isActive)
                    print('load firebase disable user in all:', image.get_name())
                    arr_disable_name.append(user.name)
                    count_disable = count_disable + 1

            except:
                print('Error disable user link: ', image.img_link)
        print('Success Load all', count_user, 'users')
        print('Success Load all', count_relative, 'Relatives')
        print('Success Load all', count_disable, 'Disable user')

        return list_known_person
    except:
        return list_known_person


# return list of Person


def encode_known_face_from_firebase_by_building(building_id):
    # init list person with their encode face

    list_known_person = {
        'personList': [],
        'faceEncoding': [],
        'faceName': [],
        'faceRelative': [],
        'faceisActive':[]
    }
    try:
        x_api = (
            'http://localhost:8888/api/User/FaceImages?BuildingId=' + str(building_id))
        y_api = (
            'http://localhost:8888/api/Relative/FaceImages?BuildingId=' + str(building_id))

        # fetch list user face image link from firebase
        x = requests.get(x_api.replace(' ', ''))
        result_x = x.json()
        urls_x = result_x['ImageURLs']

        # fetch list relatives face image link from firebase
        y = requests.get(y_api.replace(' ', ''))
        result_y = y.json()
        urls_y = result_y['ImageURLs']

        arr_user_img = []
        arr_relative_img = []

        # load all image from url
        if(len(urls_x) > 0):
            arr_user_img = load_images_from_firebase(urls_x, False)
            print('================================ user img in building ',building_id,': ', len(arr_user_img),'================================')
        if(len(urls_y) > 0):
            arr_relative_img = load_images_from_firebase(urls_y, True)
            print('================================list url relative img in building ',
                  building_id, ": ", len(arr_relative_img),'================================')

        # train all face in list
        count_user = 0
        count_relative = 0

        for image in arr_user_img:
            try:
                # Load image from image path and encode to store face


                if (len(image.user_face_encoding) != 0):
                    user = Person(image.get_name(), image.user_face_encoding, False, True)
                    list_known_person['personList'].append(user)
                    list_known_person['faceEncoding'].append(
                        user.encoding_face)
                    list_known_person['faceName'].append(user.name)
                    list_known_person['faceRelative'].append(user.isRelative)
                    list_known_person['faceisActive'].append(user.isActive)
                    print('load firebase user in Building ',
                          building_id, ': ', image.get_name())
                    count_user = count_user + 1
            except:
                print('Error link: ', image.img_link)

        for image in arr_relative_img:
            try:
                # Load image from image path and encode to store face

                if (len(image.user_face_encoding) != 0):
                    user = Person(image.get_name(), image.user_face_encoding, True, True)
                    list_known_person['personList'].append(user)
                    list_known_person['faceEncoding'].append(
                        user.encoding_face)
                    list_known_person['faceName'].append(user.name)
                    list_known_person['faceRelative'].append(user.isRelative)
                    list_known_person['faceisActive'].append(user.isActive)
                    print('load firebase relative in building',
                          building_id, ' :', image.get_name())
                    count_relative = count_relative + 1

            except:
                print('Error link: ', image.img_link)

        print('Success Load', count_user, 'User in building ', building_id)
        print('Success Load', count_relative, 'Relatives')

        return list_known_person
    except:
        return list_known_person
# TODO: THIẾU create | delete by building.

# params arr: list of new image firebase
# params isRelative
# return list of Person


def encode_new_face(arr_known_encode_face, arr, isRelative):
    # init list person with their encode face

    

    count = 0
    arr_user_img = arr
    # if(len(arr) > 0):
    #     arr_user_img = load_images_from_firebase(arr, isRelative)
    #     print('list url user img: ', len(arr_user_img))

    for image in arr_user_img:
        try:
            # Load image from image path and encode to store face
            # response = ur.urlopen(image.img_link)
            if (len(image.user_face_encoding) != 0):
                user = Person(image.get_name(), image.user_face_encoding, isRelative,True)
                # arr_known_encode_face.append(user)
                arr_known_encode_face['personList'].append(user)
                arr_known_encode_face['faceEncoding'].append(
                    user.encoding_face)
                arr_known_encode_face['faceName'].append(user.name)
                arr_known_encode_face['faceRelative'].append(user.isRelative)
                arr_known_encode_face['faceisActive'].append(user.isActive)
                print('encode new face :', image.get_name())
                count = count+1
        except:
            print('Error link: ', image.img_link)
    print('Success Load', count, 'New Faces')


def remove_know_face_by_username(username, arr_known_face):
    i = 0
    while i < len(arr_known_face['personList']):
        if len(arr_known_face['personList']) == 0:
            break
        if (arr_known_face['faceRelative'][i]) and (arr_known_face['personList'][i].name in username):
            del arr_known_face['personList'][i]
            del arr_known_face['faceEncoding'][i]
            del arr_known_face['faceName'][i]
            del arr_known_face['faceRelative'][i]
            del arr_known_face['faceisActive'][i]
        else :
            if arr_known_face['personList'][i].name+'@' in username:
                del arr_known_face['personList'][i]
                del arr_known_face['faceEncoding'][i]
                del arr_known_face['faceName'][i]
                del arr_known_face['faceRelative'][i]
                del arr_known_face['faceisActive'][i]
            else:
                i = i + 1

def disable_know_face_by_username(username, arr_known_face, isActive):
    i = 0
    while i < len(arr_known_face['personList']):
        if len(arr_known_face['personList']) == 0:
            break
        if arr_known_face['personList'][i].name+'@' in username:
            arr_known_face['faceisActive'][i] = isActive
        i = i + 1


# Face detect


def detect_face_location(camera_image, arr_known_face):
    arr_result = []
    arr_person = []

    height, width = camera_image.shape[:-1]

    img = camera_image[:, :, ::-1]

    unknown_image = img

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image,
                                                     face_locations)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    if (len(face_locations) > 0):
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            is_relative = None
            is_active = None
            if(len(arr_known_face['faceEncoding']) > 0):
                face_distances = face_recognition.face_distance(
                    arr_known_face['faceEncoding'], face_encoding)
                best_match_index = np.argmin(face_distances)

                if face_distances[best_match_index] <= tolerance:
                    name = arr_known_face['faceName'][best_match_index]
                    is_relative = arr_known_face['faceRelative'][best_match_index]
                    is_active = arr_known_face['faceisActive'][best_match_index]
                    arr_person.append(name)
                else:
                    arr_person.append(name)
            else:
                arr_person.append(name)
            result = {
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left,
                "code": name,
                "isRelative": is_relative,
                "isActive": is_active
            }
            arr_result.append(result)

    shape = {"height": height, "width": width}
    result = {"shape": shape, "info": arr_result, "people": arr_person}
    return result

# roll pitch yaw


def check_one_person_in_frame(image):
    img = image[:, :, ::-1]

    unknown_image = img

    face_locations = face_recognition.face_locations(unknown_image)
    if(len(face_locations) == 1):
        return True
    else:
        return False


def preprocessHeadPoseV2():
    x, y = pkl.load(open(face_pose_pkl_path, 'rb'))
    roll, pitch, yaw = y[:, 0], y[:, 1], y[:, 2]

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_test,
                                                    y_test,
                                                    test_size=0.5,
                                                    random_state=42)

    std = StandardScaler()
    std.fit(x_train)
    x_train = std.transform(x_train)
    x_val = std.transform(x_val)
    x_test = std.transform(x_test)
    model = load_model(face_pose_model_path)
    return [std, model]


def processHeadPoseV2(image, predictor, detector, arr):

    std, model = arr
    face_rect = detector(image, 1)
    img = image[:, :, ::-1]
    top, right, bottom, left = 0, 0, 0, 0
    unknown_image = img
    if len(face_rect) != 1:
        return ({"headpose": [1000, 1000, 1000]})

    face_locations = face_recognition.face_locations(unknown_image)
    if(len(face_locations) != 1):
        return ({"headpose": [1000, 1000, 1000]})

    else:
        top, right, bottom, left = face_locations[0]
    dlib_points = predictor(image, face_rect[0])
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        face_points.append(np.array([x, y]))
    assert (len(face_points) == 68), "len(face_points) must be 68"
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i + 1, 68):
            features.append(np.linalg.norm(face_points[i] - face_points[j]))
    features = np.array(features).reshape(1, -1)
    features = std.transform(features)
    y_pred = model.predict(features)
    roll_pred, pitch_pred, yaw_pred = y_pred[0]
    headpose = [str(yaw_pred), str(pitch_pred), str(roll_pred)]
    result = {"headpose": headpose,
              "info": {"top": top,
                       "right": right,
                       "bottom": bottom,
                       "left": left}
              }

    return result
