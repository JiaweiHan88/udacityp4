# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

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
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model makes use of the convolutional neural network proposed by Nvidia in "End to End Learning for Self-Driving Cars", 
 which is visualized in the following diagram:
 
The code can be found in (model.py lines 18-24). 

The model includes RELU activation functions to introduce nonlinearity, 
and the data is normalized in the model using a Keras lambda layer (code line 18). Its a simple normalization method to normalize the pixel data to -0.5 to 0.5.
In addition, several dropout layers have been added.

#### 2. Attempts to reduce overfitting in the model

In order to prevent overfitting, we mostly expand our dataset and use preprocessing to generalize our dataset.
First of all, our dataset consists of data from both directions of driving and also recovery driving. In addition we flip all data horizontally to add more data.
We added two dropout layer after the first two convolutional layers and one dropout layer in front of the fully connected layer. (Apparently for this project, with good training data and model, overfitting doesnt seem to be a problem, since )
The model was trained and validated on different data sets to ensure that the model was not overfitting. We split 20 % of our dataset to form the validation set.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, we decrease the default learning rate to 0.0005

#### 4. Appropriate training data

I used a combination of center lane driving, change of direction and recovery laps, where i drove first to the outer side and then started recording the process to recovering to the center.
Recovery was only done for the outer edge, because i could observe an understeering in sharp corners, whereas the car never oversteered and hit the inner edge.
I also drove multiple times across curves, because the main part of the track are straight, so we want to gather more data on curves to have a more uniform distribution.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find a suitable base network model and then adapt it for our use.
The model can be verified using the simulator and depending on the loss values and simulator results adapted to better fit for our usage.

My first step was to use a convolution neural network model proposed by Nvidia, because the purpose of that network was similar to our project.
They also use a front facing camera and try to map raw pixels to a steering command using a CNN.

I first implemented the same model and adapted the input shape to fit our data.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
The initial implementation was already promising in delivering good results on both training and validation loss, so i tested the initial model with only a normalization preprocessing on the example data set provided by the workspace.
The main issue i saw was driving through sharp corners where the car would hit the outer border and stuttering while driving.

To improve the performance the following steps were taken:

- convert to YUV color space
- crop out a bit more image lines
- resize to the same input shape used in NVidia paper (i assume they optimized their filter to their input dimension)
- flip the whole data set in order to have more data and reduce overfitting and bias to left steering
- experimenting dropout layers to further reduce overfitting
- experimenting with weight regularization

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track at sharp corners. To improve the driving behavior in these cases, I added training data with recovery laps and also more data of corner driving.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Note: During experimenting with dropout layers and weight regularization using l2 regularizer, i could see that validation loss and training loss were both decreasing and reach almost the same value. On test track one, there was no significant improvement though, even without dropout and weight regularization, the model was able to drive the whole track.
On test track two i could see some improvements, but the significance of training data still overweights. Because i recorded only two laps of normal driving, i still have issues on some of the sharp corners, which can not be resolved by tuning the model. I am sure that more data on track two and some data augmentation will lead to even better results (since track two has different lighting and slops, we should use random brightness and vertical shift and random transformation to better generalize the data and make the model more robust)

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda (Lambda)              (None, 66, 200, 3)        0
_________________________________________________________________
conv2d (Conv2D)              (None, 31, 98, 24)        1824
_________________________________________________________________
dropout (Dropout)            (None, 31, 98, 24)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 47, 36)        21636
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 47, 36)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 5, 22, 48)         43248
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 20, 64)         27712
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 18, 64)         36928
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0
_________________________________________________________________
dropout_2 (Dropout)          (None, 1152)              0
_________________________________________________________________
dense (Dense)                (None, 100)               115300
_________________________________________________________________
dense_1 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_2 (Dense)              (None, 10)                510
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________
```

The visualization of the model was presented in the image above.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded ~ two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the outer border of the road back to center so that the vehicle would learn to recover to the center at curves. Because we had a lot of center driving data, the vehicle will never hit any border in low curvature situations.
 These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would generalize our data better. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

For real world camera data, i would also add some random image transformations like in the traffic sign classifier (random brightness, scaling, translation, perspective transformation etc)
For the simulator since the camera is producing "perfect" data and the environmental parameters stays the same in testtrack 1, I left the augmentation part out because i didn't expect a big improvement for our usecase. (except for horizontal flip, because our track are bias to one direction and the flipping basically adds another track to our dataset)

After the collection process, I had X number of data points. I then preprocessed this data by converting to YUV as i did in the traffic sign project, which delivered good results.
Then i cropped out the top x and bottom x image lines, because they do not provide any helpful information for steering angle.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer with a decreased learning rate of xxx.

Note:
After the traffic sign classifier and behavior clone project i came to the conclusion that after an appropriate model was found (we can make use of existing models), the most important factor on the training results are training data as well as data augmentation and processing.
Tuning on filter size and other hyperparameter may lead to small improvements in validation accuracy, but especially in this project, training data are vital for the model to recognize all driving situations. E.g. if the car wanders off to the side and our training data doesn't contain data to recover to the center, the network will fail to produce useful results.
If we have not enough data on sharp corners, the car will most likely understeer and crash to the outer border. These issues can not be fixed by tuning the model or hyperparameters (just a bit by reducing overfitting).
