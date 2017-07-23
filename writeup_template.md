# Behavioral Cloning

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model3.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* convert_filepath.py: convert the self generated training data's file path in the "driving_log.csv" file.
  * run: python convert_filepath.py datapath; 
  * i.e. if the data are under **'data'** folder: 
    ```
    python convert_filepath.py data/
    ```

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model3.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural networks with RELU activation (model.py lines 96-100).
```
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
```

The data is normalized and Centralize in the model using a Keras lambda layer (code line 94). 
```
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
```
The image data is cropped in the model using a Keras Cropping2D layer (code line 90). 
```
model.add(Cropping2D(cropping=((70,25),(1,1)),input_shape=(160,320,3)))
```
The image data is resized to (64,64) in the model using a Keras lambda layer (code line 92). 
```
model.add(Lambda(resize_img))

def resize_img(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (64, 64))
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 101,104). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 34-35). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 111).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.


For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
Some ideas before the start:
 * Start with simple architecure.
 * We need to process image data, so the best solution should include convolutional layers.
 * Driving the car in the simulator, we just need to detect edges and lanes. Therefore, do not add too many convolutional layers.
 * Since we need to continuesly predict steering values, it is regression problem. Therefore, we need to use MSE loss function. And we do not use SOFTMAX at the end of architecture.
 
The overall strategy for deriving a model architecture:
 * Start with simple architecure similiar to the LeNet.
 * Adapt Adam optimizer so that we do not need to worry about tuning the learning rate
 * Examine the loss, if it is not decreasing significantly, we need to add more convolutional or dense layers. High loss means the architeture is poor on learning.
 * If training loss is decreasing but validation loss is not, we need to add dropout layers. 

My first step was to use a convolution neural network model, the NVIDIA Architecture. I thought this model might be appropriate because:
 * it is industry proven architecture for automonous driving
 * it has relatively few CNN layers which should work better for detecting edges and lanes.
 * it is light and fast. It takes less process time. So that the model is able to predict steering on time. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
```
Epoch 1/7
38572/38572 [==============================] - 69s - loss: 0.0173 - val_loss: 0.0179
Epoch 2/7
38572/38572 [==============================] - 65s - loss: 0.0145 - val_loss: 0.0180
Epoch 3/7
38572/38572 [==============================] - 65s - loss: 0.0134 - val_loss: 0.0173
Epoch 4/7
38572/38572 [==============================] - 65s - loss: 0.0124 - val_loss: 0.0179
Epoch 5/7
38572/38572 [==============================] - 65s - loss: 0.0116 - val_loss: 0.0209
Epoch 6/7
38572/38572 [==============================] - 65s - loss: 0.0109 - val_loss: 0.0184
Epoch 7/7
38572/38572 [==============================] - 65s - loss: 0.0101 - val_loss: 0.0192

```

To combat the overfitting, I modified the model to add dropout layers. Then the result looks much better.
```
Epoch 1/7
38568/38568 [==============================] - 33s - loss: 0.0245 - val_loss: 0.0181
Epoch 2/7
38568/38568 [==============================] - 31s - loss: 0.0186 - val_loss: 0.0168
Epoch 3/7
38568/38568 [==============================] - 32s - loss: 0.0174 - val_loss: 0.0161
Epoch 4/7
38568/38568 [==============================] - 31s - loss: 0.0166 - val_loss: 0.0157
Epoch 5/7
38568/38568 [==============================] - 31s - loss: 0.0159 - val_loss: 0.0155
Epoch 6/7
38568/38568 [==============================] - 31s - loss: 0.0153 - val_loss: 0.0154
Epoch 7/7
38568/38568 [==============================] - 31s - loss: 0.0147 - val_loss: 0.0153
```

The final step was to run the simulator to see how well the car was driving around track one. The vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes: 
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| (160,320,3) 							| 
| Cropping2D    		| cropping=((70,25),(1,1))  							| 
| Lambda(resize_images)   		| output = (64, 64) 							| 
| Lambda(normalize&Centralize) 	| 				| 
| Convolution2D  24, 5x5  | subsample=(2,2), activation="relu" 	|
| Convolution2D  36, 5x5  | subsample=(2,2), activation="relu" 	|
| Convolution2D  48, 5x5  | subsample=(2,2), activation="relu" 	|
| Convolution2D  64, 3x3  | activation="relu" 	|
| Convolution2D  64, 3x3  | activation="relu" 	|
| Dropout      	| 0.3	|
| Flatten      	| 	|
| Dense		| 100		|
| Dropout      	| 0.2	|
| Dense		| 50		|
| Dense		| 10		|
| Dense		| 1		|
|						|												|

#### 3. Creation of the Training Set & Training Process

It is hard control the simulator actually. Following tips helped in controlling the vehicle: choose the lowest Screen Resolution and fastest Graphics quality. And Use mouse steer which seems better for me.

 ![configuration](/assets/graphics.PNG)
 
Here are images from left,center and right cameras:

 ![alt text](/assets/images.PNG)

To let the model learn better, I cropped each image. 
 ![alt text](/assets/cropped.PNG)

To make the training process go faster, I resized each image. 
 ![alt text](/assets/resized.PNG)


After the collection process, I had 8036 number of data points. I randomly shuffled the data set and put 20% the data into a validation set. In order to make more data to generalize the model, I adopted images from left and right cameras. In addition I flipped each image. Following summary shows the data increment.
```
Total Training Samples:  6428
Total Validation Samples:  1608
Total Training Samples after preprocessing:  38568
Total Validation Samples after preprocessing:  9648
```
I used an adam optimizer so that manually training the learning rate wasn't necessary.

I used generator to overcome the memory shortage problem.
