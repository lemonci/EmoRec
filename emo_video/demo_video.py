#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2 as cv
cap = cv.VideoCapture('200.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    if cv.waitKey(0) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


# In[1]:


### General imports ###
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from time import sleep
import re
import os
import argparse
from collections import OrderedDict
import matplotlib.animation as animation

### Image processing ###
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage
import cv2
import dlib
from __future__ import division
from imutils import face_utils

### CNN models ###
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.utils import np_utils
from keras.regularizers import l2#, activity_l2
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras import models
from keras.utils.vis_utils import plot_model
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras import layers

### Build SVM models ###
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

### Same trained models ###
import h5py
from keras.models import model_from_json
import pickle


# In[2]:


path = '/home/monica/EmoRec/03-Video/'
local_path = '/home/monica/EmoRec/03-Video/'


# In[3]:


shape_x = 48
shape_y = 48


# In[4]:


#Defining labels 
def get_label(argument):
    labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad' , 5:'Surprise', 6:'Neutral'}
    return(labels.get(argument, "Invalid emotion"))


# In[5]:


def detect_face(frame):
    
    #Cascade classifier pre-trained model
    cascPath = '/home/monica/anaconda3/envs/EmoRec/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    #BGR -> Gray conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Cascade MultiScale classifier
    detected_faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6,
                                                  minSize=(shape_x, shape_y),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
    
    coord = []
    
    for x, y, w, h in detected_faces :
        if w > 100 :
            sub_img=frame[y:y+h,x:x+w]
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
            coord.append([x,y,w,h])
    
    return gray, detected_faces, coord


# In[6]:


#Extraire les features faciales
def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
    gray = faces[0]
    detected_face = faces[1]
    
    new_face = []
    
    for det in detected_face :
        #Region dans laquelle la face est détectée
        x, y, w, h = det
        #X et y correspondent à la conversion en gris par gray, et w, h correspondent à la hauteur/largeur
    
        #Offset coefficient, np.floor takes the lowest integer (delete border of the image)
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray transforme l'image
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
    
        #Zoom sur la face extraite
        new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))
        #cast type float
        new_extracted_face = new_extracted_face.astype(np.float32)
        #scale
        new_extracted_face /= float(new_extracted_face.max())
        #print(new_extracted_face)
    
        new_face.append(new_extracted_face)
    
    return new_face


# In[ ]:


def entry_flow(inputs) :
    
    x = Conv2D(32, 3, strides = 2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(64,3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    previous_block_activation = x
    
    for size in [128, 256, 728] :
    
        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)
    
        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        x = MaxPooling2D(3, strides=2, padding='same')(x)
        
        residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)
        
        x = keras.layers.Add()([x, residual])
        previous_block_activation = x
    
    return x


# In[ ]:


def middle_flow(x, num_blocks=8) :
    
    previous_block_activation = x
    
    for _ in range(num_blocks) :
    
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
    
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
        
        x = keras.layers.Add()([x, previous_block_activation])
        previous_block_activation = x
    
    return x


# In[ ]:


def exit_flow(x, num_classes=7) :
    
    previous_block_activation = x
    
    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
    x = keras.layers.Add()([x, residual])
      
    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    return x


# In[ ]:


inputs = Input(shape=(shape_x, shape_y, 1))
outputs = exit_flow(middle_flow(entry_flow(inputs)))


# In[ ]:


xception = Model(inputs, outputs)


# In[ ]:


#plot_model(xception, to_file='model_plot_4.png', show_shapes=True, show_layer_names=True)


# In[ ]:


#xception.summary()


# In[58]:


with open(local_path + 'savedmodels/xception2.json','r') as f:
    json = f.read()
model = model_from_json(json)

model.load_weights(local_path + 'savedmodels/xception-33-0.66.h5')
print("Loaded model from disk")


# In[ ]:


layer_outputs = [layer.output for layer in model.layers[:12]] 
# Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)


# In[ ]:


layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
images_per_row = 16


# In[ ]:


trump = '/home/monica/EmoRec/03-Video/Images/Test_Images/trump.jpg'
trump_face = cv2.imread(trump)
face = extract_face_features(detect_face(trump_face))[0]

to_predict = np.reshape(face.flatten(), (1,48,48,1))
res = model.predict(to_predict)
activations = activation_model.predict(to_predict)


# In[ ]:


for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


# In[ ]:





# In[ ]:


#image
'''
filename = 'fear2'
image = '/home/monica/EmoRec/03-Video/Images/Test_Images/' + filename + '.jpg'
frame = cv2.imread(image)
for face in extract_face_features(detect_face(frame)) :
    to_predict = np.reshape(face.flatten(), (1,48,48,1))
    res = model.predict(to_predict)
    result_num = np.argmax(res)
    print(result_num)'''


# In[7]:


#image
face_index = 0
gray, detected_faces, coord = detect_face(frame)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face_detect = dlib.get_frontal_face_detector()
predictor_landmarks = dlib.shape_predictor("/home/monica/EmoRec/03-Video/Models/Landmarks/face_landmarks.dat")
rects = face_detect(gray, 1)
#try: 
for (i, rect) in enumerate(rects):
    shape = predictor_landmarks(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Identify face coordinates
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    face = gray[y:y+h,x:x+w]

    #Zoom on extracted face
    face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))

    #Cast type float
    face = face.astype(np.float32)

    #Scale
    face /= float(face.max())
    face = np.reshape(face.flatten(), (1, 48, 48, 1))

    #Make Prediction
    prediction = model.predict(face)
    prediction_result = np.argmax(prediction)
            # Rectangle around the face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#    cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (j, k) in shape:
        cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)

   # 12. Add prediction probabilities
#    cv2.putText(frame, "----------------",(x+w-10,100 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
    if prediction_result == 0:
        cv2.putText(frame, "Angry : " + str(round(prediction[0][0],3)),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif prediction_result == 1 :
        cv2.putText(frame, "Disgust : " + str(round(prediction[0][1],3)),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif prediction_result == 2 :
        cv2.putText(frame, "Fear : " + str(round(prediction[0][2],3)),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif prediction_result == 3 :
        cv2.putText(frame, "Happy : " + str(round(prediction[0][3],3)),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif prediction_result == 4 :
        cv2.putText(frame, "Sad : " + str(round(prediction[0][4],3)),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif prediction_result == 5 :
        cv2.putText(frame, "Surprise: " + str(round(prediction[0][4],3)),(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else :
        cv2.putText(frame, "Neutral : " + str(round(prediction[0][6],3)),(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#cv2.putText(frame,'Number of Faces : ' + str(len(rects)),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)
#cv2.imshow('image',frame)
cv2.imwrite(filename +'_result.jpg', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


# In[9]:


def print_result(image):
    frame = cv2.imread(image)
    face_index = 0
    gray, detected_faces, coord = detect_face(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detect = dlib.get_frontal_face_detector()
    predictor_landmarks = dlib.shape_predictor("/home/monica/EmoRec/03-Video/Models/Landmarks/face_landmarks.dat")
    rects = face_detect(gray, 1)
    #try: 
    for (i, rect) in enumerate(rects):
        shape = predictor_landmarks(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Identify face coordinates
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = gray[y:y+h,x:x+w]

        #Zoom on extracted face
        face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))

        #Cast type float
        face = face.astype(np.float32)

        #Scale
        face /= float(face.max())
        face = np.reshape(face.flatten(), (1, 48, 48, 1))

        #Make Prediction
        prediction = model.predict(face)
        prediction_result = np.argmax(prediction)
        
        y_label.append(emotion_labels[label])
        y_predict.append(prediction_result)
        #print(image, prediction_result)
        #if prediction_result != 0:
        #    print(image, prediction_result)


# In[59]:


y_label = []
y_predict = []
emotion_labels = {
  "Angry": 0,
  "Disgust": 1,
  "Fear": 2,
  "Happy": 3,
  "Sad": 4,
  "Surprise": 5,
  "Neutral": 6
}


# In[60]:


#fer2013.emotion: Angry:0,  Disgust:1,  Fear:2,  Happy:3,  Sad:4,  Surprise:5,  Neutral:6
for label in emotion_labels.keys():
    for image in absoluteFilePaths("/home/monica/EmoRec/03-Video/SFEW_dataset/SFEW_2/Val/"+label):
        print_result(image)


# In[43]:





# In[44]:





# In[100]:





# In[117]:


'''from collections import Counter
print(Counter(y_label))
print(Counter(y_predict))'''


# In[61]:


'''from sklearn.metrics import confusion_matrix

labels = range(7)
cm = confusion_matrix(y_label, y_predict, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
#plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticks(range(0,7))
ax.set_yticks(range(0,7))
ax.set_xticklabels(['Angry', 'Disgust', 'Fear', 'Happy',  'Sad',  'Surprise',  'Neutral'])
ax.xaxis.tick_bottom()
ax.set_yticklabels(['Angry', 'Disgust', 'Fear', 'Happy',  'Sad',  'Surprise',  'Neutral'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()'''


# In[40]:


#video
face_detect = dlib.get_frontal_face_detector()
predictor_landmarks = dlib.shape_predictor("/home/monica/EmoRec/03-Video/Models/Landmarks/face_landmarks.dat")

filename = '1620330000759667.mp4'

video_capture = cv.VideoCapture('/home/monica/EmoRec/03-Video/Test_video/'+filename)
flag = 0
j = 1
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv.VideoWriter('/home/monica/EmoRec/03-Video/Test_video/result_' + filename, fourcc, 12.0, (int(video_capture.get(3)),int(video_capture.get(4))))


while video_capture.isOpened():
    ret, frame = video_capture.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Capture frame-by-frame
    
    face_index = 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detect(gray, 1)
    #gray, detected_faces, coord = detect_face(frame)
    
    for (i, rect) in enumerate(rects):
        try:
            shape = predictor_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y+h,x:x+w]

            #Zoom sur la face extraite
            face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
            #cast type float
            face = face.astype(np.float32)
            #scale
            face /= float(face.max())
            face = np.reshape(face.flatten(), (1, 48, 48, 1))
            prediction = model.predict(face)
            prediction_result = np.argmax(prediction)

            # Rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            x_offset = -30
            y_offset = 10
            font_scale = 0.5
            for (j, k) in shape:
                cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)
            if prediction_result == 0:
                cv2.putText(frame, "Angry : " + str(round(prediction[0][0],3)),(x-x_offset,y-y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            elif prediction_result == 1 :
                cv2.putText(frame, "Disgust : " + str(round(prediction[0][1],3)),(x-x_offset,y-y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            elif prediction_result == 2 :
                cv2.putText(frame, "Fear : " + str(round(prediction[0][2],3)),(x-x_offset,y-y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            elif prediction_result == 3 :
                cv2.putText(frame, "Happy : " + str(round(prediction[0][3],3)),(x-x_offset,y-y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            elif prediction_result == 4 :
                cv2.putText(frame, "Sad : " + str(round(prediction[0][4],3)),(x-x_offset,y-y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            elif prediction_result == 5 :
                cv2.putText(frame, "Surprise: " + str(round(prediction[0][4],3)),(x-x_offset,y-y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            else :
                cv2.putText(frame, "Neutral : " + str(round(prediction[0][6],3)),(x-x_offset,y-y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        except:
            continue
            '''
         # 12. Add prediction probabilities
        cv2.putText(frame, "Emotional report : ",(40,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        cv2.putText(frame, "Angry : " + str(round(prediction[0][0],3)),(40,140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        cv2.putText(frame, "Disgust : " + str(round(prediction[0][1],3)),(40,160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        cv2.putText(frame, "Fear : " + str(round(prediction[0][2],3)),(40,180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(frame, "Happy : " + str(round(prediction[0][3],3)),(40,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(frame, "Sad : " + str(round(prediction[0][4],3)),(40,220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(frame, "Surprise : " + str(round(prediction[0][5],3)),(40,240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        cv2.putText(frame, "Neutral : " + str(round(prediction[0][6],3)),(40,260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
        
        # draw extracted face in the top right corner
        #frame[face_index * shape_x: (face_index + 1) * shape_x, -1 * shape_y - 1:-1, :] = cv2.cvtColor(face * 255, cv2.COLOR_GRAY2RGB)

        # 13. Annotate main image with a label
        if prediction_result == 0 :
            cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction_result == 1 :
            cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction_result == 2 :
            cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction_result == 3 :
            cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction_result == 4 :
            cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif prediction_result == 5 :
            cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else :
            cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
       
        # 5. Eye Detection and Blink Count
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Compute Eye Aspect Ratio
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
            
        # And plot its contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Compute total blinks and frequency
        if ear < thresh:
            flag += 1
            #cv2.putText(frame, "Blink", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        
        cv2.putText(frame, "Total blinks : " + str(flag), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)
        #cv2.putText(frame, "Blink Frequency : " + str(int(flag/j)), (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
        
        # 6. Detect Nose
        nose = shape[nStart:nEnd]
        noseHull = cv2.convexHull(nose)
        cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

        # 7. Detect Mouth
        mouth = shape[mStart:mEnd]
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            
        # 8. Detect Jaw
        jaw = shape[jStart:jEnd]
        jawHull = cv2.convexHull(jaw)
        cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
            
        # 9. Detect Eyebrows
        ebr = shape[ebrStart:ebrEnd]
        ebrHull = cv2.convexHull(ebr)
        cv2.drawContours(frame, [ebrHull], -1, (0, 255, 0), 1)
        ebl = shape[eblStart:eblEnd]
        eblHull = cv2.convexHull(ebl)
        cv2.drawContours(frame, [eblHull], -1, (0, 255, 0), 1)'''
            
    #cv2.putText(frame,'Number of Faces : ' + str(i+1),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)
    j = j + 1
    out.write(frame)
#    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:




