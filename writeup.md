# **Traffic Sign Recognition** 

[//]: # (Image References)

[visualization]: ./visualization.png "Visualization"
[validation_bar]: ./validation_bar.png "Validation Bar"
[training_bar]: ./training_bar.png "Training Bar"
[testing_bar]: ./testing_bar.png "Testing Bar"
[grayscaling]: ./grayscaling.png "Grayscaling"
[minimax]: ./minimax.png "Minimax"
[softmax]: ./softmax.png "Softmax"

[aug_example1]: ./aug_example1.png "Aug Example 1"
[aug_example2]: ./aug_example2.png "Aug Example 2"

[image1]: ./validation_new/1.jpg "Traffic Sign 1"
[image2]: ./validation_new/18.jpg "Traffic Sign 2"
[image3]: ./validation_new/23.jpg "Traffic Sign 3"
[image4]: ./validation_new/25.jpg "Traffic Sign 4"
[image5]: ./validation_new/31.jpg "Traffic Sign 5"
## Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/romkof/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1.A basic summary of the data set. In the code.

I used python and the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

#### 2.Visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training validation and testing data are distributed

![alt text][visualization]
![alt text][training_bar]
![alt text][validation_bar]
![alt text][testing_bar]

### Design and Test a Model Architecture

#### 1.Preprocessed the image data.

As a first step, I decided to generate additional data because some sign have to small number of examples, and network will be bias about classes, that presented in dataset the most.

To add more data to the the data set, I used the great library  [imgaug](https://github.com/aleju/imgau) . I used different techniques like cropping, flipping, contrast normalization, superpixeling and others.

Here is an examples augmented images:

![alt text][aug_example1]
![alt text][aug_example2]

I decided to convert the images to grayscale because for signs classification color is not important information. Grayscaling images will help to reduce network training time. 

Here is an example of a traffic sign image  after grayscaling.

![alt text][grayscaling]

As a last step, I normalized the image data using Minimax optimization, because it help optimizer to perform better, having all values zero mean and equal variance.

 
Exploratory visualization The difference between the original data set and the augmented data set is the following:
 - all classes have the same amout of examples
 - amount of  most popular examples is increased by 35 %


#### 2.Final model architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x8 				    |
| Convolution 1x1	    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 		     		|
| Fully connected		| input = 800, output = 400          			|
| RELU					|												|
| Dropout               | keep probability 0.7                          |
| Fully connected		| input = 400, output = 100          			|
| RELU					|												|
| Dropout               | keep probability 0.7                          |
| Softmax				| input = 400, output = 43        				|
|						|												|

 


#### 3. Hyperparameters 

To train the model, I used an using learning rate = 0.001
and EPOCHS = 20, batch size was 128 adn droupout value was 0.7


#### 4. Describe solution

My final model results were:
* training set accuracy of  1.000
* validation set accuracy of 0.961
* test set accuracy of 0.942

Iterative approach was chosen for training the network, first architecture was chosen LeNet-5, because it is pretty simple and small network. This architecture was adjusted with making more deeper hidden layer and adding droupout.In future i would like to test the same datasets on bigger networks like VGG-16 and Xception networks. Adjusting an architecture was done be due to very high overfitting. Accuracy value was 1.0 just after 5 epochs. Despite data augmentation and high dropout value, current network has signs of overfitting.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The first image might be difficult to classify because ...

#### 2. Model's predictions 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild animals crossing | Wild animals crossing  						| 
| General caution       | General caution  								|
| Slippery road			| Slippery road									|
| Road work	      		| Road work			        	 				|
| Speed limit (30km/h)	| Speed limit (30km/h)      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.942

#### 3.Top 5 softmax probabilities

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

For the all images, except for slippery roads sing, the model is sure what was displayed. Here is prediction for all 5 images:

![alt text][softmax]




