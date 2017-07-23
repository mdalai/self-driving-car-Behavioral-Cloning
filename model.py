import sys
import csv
import cv2
import numpy as np

# DATA FOLDER PATH can be input as parameter, i.e. "python model.py data/data/"
# by no parameter is given, DATAPATH will be set to "data/"
if len(sys.argv) == 1:
    DATAPATH = "data/"
else:
    DATAPATH=sys.argv[1]

lines = []
with open(DATAPATH + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Check if the image filepath is correct in the driving_log.csv.
# Self generated training data has full path and joined by '\'. This has to be updated first.
# I have developed convert_filepath.py to update the csv file.
if lines[1][0].find('\\') != -1:
    print('Need to UPDATE the filepath in "driving_log.csv" !!!')
    sys.exit()



# Exclude the header
lines = lines[1:]
print("Total System Generated Data: ", len(lines))

####### Data Pre-processing #################################
## 1. SPLIT INTO train and validation data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print("Total Training Samples: ", len(train_samples))
print("Total Validation Samples: ", len(validation_samples))

## 2. Get More training data: by doing following steps, 6 times more data can be generated.
###  2.1 utilize center, left, right cemaras' images
###  2.2 flip each image

def allCameras_and_augment(line):

    # 1. utilize center, left, right cemaras' images
    images, steerings=[], []
    for i in range(3):
        #image = cv2.imread(DATAPATH + line[i])
        image = cv2.imread(DATAPATH + line[i].strip())
        images.append(image)
        steering = float(line[3])
        # left camera: turn slightly right toward the center
        if i == 1:
            steering += 0.2
        # right camera: turn slightly letf toward the center
        if i == 2:
            steering -= 0.2

        steerings.append(steering)

    # 2. augmentation generates more training data
    aug_images, aug_steerings = [],[]
    for image, steering in zip(images, steerings):
        aug_images.append(image)
        aug_steerings.append(steering)
        aug_images.append(cv2.flip(image,1))
        aug_steerings.append(steering*-1.0)

    return aug_images, aug_steerings

####### The Model #################################
###  Some of data preprocessing steps included in the Keras model. So that no need to worry about doing these preprocessing steps when autonoums driving
###  adopted the NVIDIA Architecture.

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#*** Function to resize images: adapted Tensorflow image API
#***    param: images
#***    return: resized images
def resize_img(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (64, 64))


model = Sequential()
# Crop image: top-70, bottom-20, left-1, right-1
model.add(Cropping2D(cropping=((70,25),(1,1)),input_shape=(160,320,3)))
# Resize image: 
model.add(Lambda(resize_img))
# Normalize and Centralize
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# NVIDIA Architecture
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# It is Regression:
# - no need SOFTMAX
# - loss function: MSE
model.compile(loss='mse', optimizer='adam')

####### Training #################################
## adopted python generator technology in order to solve memory shortage problem

# *** Generator Function
import sklearn
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # preprocessing: 
                #*** 1. utilize all cameras images
                #*** 2. augment: flip each image
                imgs,angls = allCameras_and_augment(batch_sample)
                for img in imgs:
                    images.append(img)
                for angl in angls:
                    angles.append(angl)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

print("Total Training Samples after preprocessing: ", len(train_samples)*6)
print("Total Validation Samples after preprocessing: ", len(validation_samples)*6)



#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, 
    validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=7)

model.save('models/model3.h5')

