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

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

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

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because:
 * it has few CNN layers which should work better for detecting edges and lanes.
 * it is simple and fast. We can get the idea of which direction to go in next step.
 * it is computational efficient. We can troubleshoot problem faster.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the final architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 64x64   							| 
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
 
To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
