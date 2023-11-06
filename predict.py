import cv2
import numpy as np
from keras.models import load_model
import torch

# Load the trained model to classify sign
model1 = load_model('traffic_sign_model_2.h5')
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/best.pt', force_reload=True)

classes = { 
    0:'No sign detected',
    1:'Speed limit (20km/h)',
    2:'Speed limit (30km/h)',
    3:'Speed limit (50km/h)',
    4:'Speed limit (60km/h)',
    5:'Speed limit (70km/h)',
    6:'Speed limit (80km/h)',
    7:'End of speed limit (80km/h)',
    8:'Speed limit (100km/h)',
    9:'Speed limit (120km/h)',
    10:'No passing',
    11:'No passing veh over 3.5 tons',
    12:'Right-of-way at intersection',
    13:'Priority road',
    14:'Yield',
    15:'Stop',
    16:'No vehicles',
    17:'Veh > 3.5 tons prohibited',
    18:'No entry',
    19:'General caution',
    20:'Dangerous curve left',
    21:'Dangerous curve right',
    22:'Double curve',
    23:'Bumpy road',
    24:'Slippery road',
    25:'Road narrows on the right',
    26:'Road work',
    27:'Traffic signals',
    28:'Pedestrians',
    29:'Children crossing',
    30:'Bicycles crossing',
    31:'Beware of ice/snow',
    32:'Wild animals crossing',
    33:'End speed + passing limits',
    34:'Turn right ahead',
    35:'Turn left ahead',
    36:'Ahead only',
    37:'Go straight or right',
    38:'Go straight or left',
    39:'Keep right',
    40:'Keep left',
    41:'Roundabout mandatory',
    42:'End of no passing',
    43:'End no passing vehicle with a weight greater than 3.5 tons' 
}

def predict_sign(frame):
    # Preprocessing the frame
    frame = np.asarray(frame)
    img = cv2.resize(frame, (30,30))
    img = np.expand_dims(img, axis=0)
    img = np.array(img)

    # Predict
    pred = model1.predict([img])[0]
    if pred.max() > 0.8:
        pred = np.argmax(pred)
    else:
        pred = -1
    sign = classes[pred+1]
    return sign

def prdict_model2(img):
    results = model2(img)   
    transformed = cv2.cvtColor(np.squeeze(results.render()), cv2.COLOR_BGR2RGB)
    cv2.imwrite('temp.jpg', transformed)
    return 0