import cv2
import numpy as np
from pathlib import Path
import time

weightPathCur_custom = str(Path().absolute()) + "/services/models/yolov4_custom.weights"
weightCfgPathCur_custom = str(Path().absolute()) + "/services/models/yolov4_custom.cfg"
net_custom = cv2.dnn.readNet(weightPathCur_custom, weightCfgPathCur_custom)

weightPathCur_full = str(Path().absolute()) + "/services/models/yolov4.weights"
weightCfgPathCur_full = str(Path().absolute()) + "/services/models/yolov4.cfg"
net_full = cv2.dnn.readNet(weightPathCur_full, weightCfgPathCur_full)


weightPathCur_tiny = str(Path().absolute()) + "/services/models/yolov4-tiny.weights"
weightCfgPathCur_tiny = str(Path().absolute()) + "/services/models/yolov4-tiny.cfg"
net_tiny = cv2.dnn.readNet(weightPathCur_tiny,weightCfgPathCur_tiny )

cocoNamePathCur_custom = str(Path().absolute()) + "/services/classes/coco_custom.names"
cocoNamePathCur_full = str(Path().absolute()) + "/services/classes/coco.names"
cocoNamePathCur_tiny = str(Path().absolute()) + "/services/classes/coco.names"




def detect_object_custom(img):
    classes = []
    with open(cocoNamePathCur_custom, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net_custom.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net_custom.getUnconnectedOutLayers()]

    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img,
                                 0.00392, (416, 416), (0, 0, 0),
                                 True,
                                 crop=False)
    net_custom.setInput(blob)
    outs = net_custom.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.7:
                # Object detected
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.4)

    # img = cv2.resize(img, None, fx=1.4, fy=1.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, h / 180, color, 2)
    img = cv2.resize(img, (width, height))
    return img[:, :, ::-1]


def detect_object_full(img):
    classes = []
    with open(cocoNamePathCur_full, "r") as f:
        classes = [line.strip() for line in f.readlines()]   
    layer_names = net_full.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net_full.getUnconnectedOutLayers()]
    
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_full.setInput(blob)
    outs = net_full.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0,255,0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, h/280, color, 2)
    img = cv2.resize(img, (width, height))
    return img[:, :, ::-1]

def detect_object_tiny(img):
    classes = []
    with open(cocoNamePathCur_tiny, "r") as f:
        classes = [line.strip() for line in f.readlines()]   
    layer_names = net_tiny.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net_tiny.getUnconnectedOutLayers()]
    
    
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_tiny.setInput(blob)
    outs = net_tiny.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    test = []
    for out in outs:
        
        for detection in out:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                
                confidences.append(float(confidence))
                class_ids.append(class_id)
                test.append({'location': [x,y,w,h],'classID':class_id})


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print('boxes:', boxes)
    print('confidences: ', confidences )
    print('index',indexes)
    font = cv2.FONT_HERSHEY_DUPLEX

    print('test', test)
    print('range:',range(len(boxes)))
    for i in range(len(boxes)):
        if i in indexes:
            print('flag 1')
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print('class id', class_ids[i])
            print('label', label)
            color = (0,255,0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, h / 280, color, 1)
    img = cv2.resize(img, (width, height))
    return img[:, :, ::-1]

def detect_object_location(img):
    classes = []
    arr_result= []
    arr_object=[]
    with open(cocoNamePathCur_tiny, "r") as f:
        classes = [line.strip() for line in f.readlines()]   
    layer_names = net_tiny.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net_tiny.getUnconnectedOutLayers()]
    
    
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_tiny.setInput(blob)
    outs = net_tiny.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        
        for detection in out:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # 0: person 67: cell-phone
    for i in range(len(boxes)):
        if i in indexes:
            if(class_ids[i] == 0):
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                result = {
                    "top": y,
                    "right":x+w,
                    "bottom": y+h,
                    "left": x,
                    "code": label
                }
                arr_object.append(label)
                arr_result.append(result)
    img = cv2.resize(img, (width, height))
    shape= {"height": height, "width":width}
    result = {"shape": shape, "info": arr_result, "objects": arr_object}
    return result

def detect_prohibited_location_custom(img):
    classes = []
    arr_result= []
    arr_object=[]
    with open(cocoNamePathCur_custom, "r") as f:
        classes = [line.strip() for line in f.readlines()]   
    layer_names = net_custom.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net_custom.getUnconnectedOutLayers()]
    
    
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_custom.setInput(blob)
    outs = net_custom.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        
        for detection in out:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    
    
    for i in range(len(boxes)):
        if i in indexes:
            if(class_ids[i] == 0):
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                result = {
                    "top": y,
                    "right":x+w,
                    "bottom": y+h,
                    "left": x,
                    "code": label
                }
                arr_object.append(label)
                arr_result.append(result)
    img = cv2.resize(img, (width, height))
    shape= {"height": height, "width":width}
    result = {"shape": shape, "info": arr_result, "objects": arr_object}
    return result



def detect_prohibited_location_full(img):
    classes = []
    arr_result= []
    arr_object=[]
    with open(cocoNamePathCur_full, "r") as f:
        classes = [line.strip() for line in f.readlines()]   
    layer_names = net_full.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net_full.getUnconnectedOutLayers()]
    
    
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net_full.setInput(blob)
    outs = net_full.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        
        for detection in out:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    
    
    for i in range(len(boxes)):
        if i in indexes:
            if(class_ids[i] == 67):
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                result = {
                    "top": y,
                    "right":x+w,
                    "bottom": y+h,
                    "left": x,
                    "code": label
                }
                arr_object.append(label)
                arr_result.append(result)
    img = cv2.resize(img, (width, height))
    shape= {"height": height, "width":width}
    result = {"shape": shape, "info": arr_result, "objects": arr_object}
    return result