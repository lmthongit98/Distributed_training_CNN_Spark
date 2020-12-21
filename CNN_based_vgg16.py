import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import cv2
import os

#Size image
IMGS = 256
#Directory paths:
main = './chest_xray/'
#categ = ['test']
categ = ['train', 'test', 'val']
classes = ['NORMAL','PNEUMONIA']
images = []
labels = []
for i in categ:
    sub_path = os.path.join(main, i)
    for j  in classes:
        path = os.path.join(sub_path, j) 
        temp = os.listdir(path)
        for x in temp:
            addr = os.path.join(path, x)
            img_arr = cv2.imread(addr)
            img_arr = cv2.resize(img_arr, (IMGS, IMGS))
            images.append(img_arr)
            if j == 'PNEUMONIA':
                l = 1
            else:
                l = 0
            labels.append(l)
print('Done.')

images = np.array(images)
labels = np.array(labels)
print(images.shape, labels.shape)

#Split train / valid
from sklearn.model_selection import train_test_split as tts

x_train, x_test, y_train, y_test = tts(images, labels, random_state = 42, test_size = .20)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_train = x_train.reshape(-1, IMGS, IMGS, 3)
x_test = x_test.reshape(-1, IMGS, IMGS, 3)
print('Done.')
print(x_train.shape, x_test.shape)

"""### Contruct Model"""
model_14 = Sequential()

# 1st Convolution block
model_14.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
model_14.add(Conv2D(16, (3, 3), activation='relu'))
model_14.add(MaxPooling2D((2, 2)))

# 2nd Convolution block
model_14.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model_14.add(Conv2D(32, (3, 3), activation='relu'))
model_14.add(MaxPooling2D((2, 2)))

# 3rd Convolution block
model_14.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model_14.add(Conv2D(64, (3, 3), activation='relu'))
model_14.add(MaxPooling2D((2, 2)))

# 4th Convolution block
model_14.add(Conv2D(96, (3, 3), dilation_rate=(2,2), activation='relu', padding='same'))
model_14.add(Conv2D(96, (3, 3), activation='relu'))
model_14.add(Conv2D(96, (3, 3), activation='relu'))
model_14.add(MaxPooling2D((2, 2)))

# 5th Convolution block
model_14.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same'))
model_14.add(Conv2D(128, (3, 3), activation='relu'))
model_14.add(Conv2D(128, (3, 3), activation='relu'))
model_14.add(MaxPooling2D((2, 2)))

# Flattened the layer
model_14.add(Flatten())

# Fully connected layers
model_14.add(Dense(64, activation='relu'))
model_14.add(Dropout(0.4))
model_14.add(Dense(1, activation='sigmoid'))

model_14.summary()

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

spark = (SparkSession
         .builder
       #.master("local[*]")
        #  .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")
         #.config('spark.executor.memory', '4g')
         #.config('spark.driver.memory', '10g')
         .getOrCreate()
)

sc = spark.sparkContext

"""### Training Model"""

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
# Compile the model.
model_14.compile(optimizer='adam', loss='binary_crossentropy',
metrics=['accuracy'])
# Build RDD from features and labels.
rdd = to_simple_rdd(sc, x_train, y_train)
# Initialize SparkModel from Keras model and Spark context.
spark_model = SparkModel(model_14, frequency='epoch', mode='asynchronous', num_workers=3)
# Train the Spark model.
spark_model.fit(rdd, epochs=10, batch_size=32, verbose=1, validation_split=0.1)

score = spark_model.master_network.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', score)

"""### Predcit and evaluate Model"""

"""### Save Model"""

import json
#lets assume 'model' is main model 
model_json = model_14.to_json()
with open("model_in_json.json", "w") as json_file:
    json.dump(model_json, json_file)


spark_model.save('spark_model.h5')

# Load saved Model and using keras to predict

from keras.models import load_model
from keras.models import model_from_json
#import json

with open('model_in_json.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model = load_model('spark_model.h5')
print("Loaded model sucessfully")

results = model.evaluate(x_test, y_test, verbose=2)
print("test loss, test acc:", results)

predictions = model.predict_classes(x_test)
predictions = predictions.reshape(1,-1)[0]

from sklearn.metrics import classification_report,confusion_matrix
#import seaborn as sns

print(classification_report(y_test, predictions, target_names = ['Normal(Class 0)','Pneumonia(Class 1)']))

cm = confusion_matrix(y_test,predictions)
print(cm)

