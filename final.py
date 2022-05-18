import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import os
from keras.preprocessing.image import load_img
from keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt
from flask import Flask , request , render_template
from PIL import Image
from io import BytesIO  
app = Flask(__name__) 

#model = load_model("/contenmodel.h5")
modeld = load_model("modeld.h5")

#dir_path = '/content/drive/MyDrive/thyroid_images/test'
@app.route('/')
def home():
    return render_template('index.html')




@app.route('/gettypeofdisease',methods=['POST'])
def gettypeofdisease():
    msge = ''
    file_pic = request.files['file']cd
    img = Image.open(file_pic)
    newsize = (200, 200)
    img = img.resize(newsize)
    #img = image.resize((200, 200))
    # plt.imshow(img)
    # plt.show()
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    #print(images)
    val = modeld.predict(images)
    if val == 0:
      msge = 'This Disease is cerospora leaf Spot'
    else:
      msge = 'This Disease is common rust'

    return render_template('index.html', msge = msge)




app.run()
