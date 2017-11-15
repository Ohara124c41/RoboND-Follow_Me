# Follow-Me Project 
___


##### Scholar: Christopher Ohara
##### Program: Udacity Robotics Nanodegree
#
## Introduction
___

##### This project consists of programming and training a Fully Convolutional Network (FCN) for a drone. The implemented hyperparameters allow the drone to track a target (the "hero') in a populated urban area. These types of problems are common in robotics and artificial intelligence, especially for autonomous navigation (self-driving cars, drones) and human-machine interaction (shop-floor robots). 
#
#
##### The target has specific characteristics that are distinguishable from environment and other people (wearing a vibrantly red shirt). This project utilizes Python, Unity, Tensorflow and Keras.
#
## Data Collection
___

##### Using the simulator, collect data for three different conditions
1) Navigate the empty map to record ambient environment
2) Spawn only the target in an otherwise unpopulated environment
3) Follow the target through the populated map
#
## Network Architecture
___
#
##### Since spatial information is required for a moving target, a fully convolutional network (FNC) is required. The amount of layers is dependent on a few factors discribed by the algorithms and layers below, but specificially, considerations need to be given to depth (increases with layers), number of hyperparameters (influenced ny separable convolutions) and methods to retain spatial information.
#
##### The image below shows the chosen network architecture for this application.
#
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/00x.jpg?raw=true)
#
## FCN Layers
___


##### In the Classroom, we discussed the different layers that constitute a fully convolutional network (FCN). The following code will introduce you to the functions that you need to build your semantic segmentation model.

### Separable Convolutions
___


##### The Encoder for your FCN will essentially require separable convolution layers, due to their advantages as explained in the classroom. The 1x1 convolution layer in the FCN, however, is a regular convolution. Implementations for both are provided below for your use. Each includes batch normalization with the ReLU activation function applied to the layers.
```
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer
```

### Bilinear Upsampling
___


The following helper function implements the bilinear upsampling layer. Upsampling by a factor of 2 is generally recommended, but you can try out different factors as well. Upsampling is used in the decoder block of the FCN.

```
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
```

## Build the Model 
___

In the following cells, you will build an FCN to train a model to detect and locate the hero target within an image. The steps are:

* Create an `encoder_block`
* Create a `decoder_block`
* Build the FCN consisting of encoder block(s), a 1x1 convolution, and decoder block(s).  This step requires experimentation with different numbers of layers and filter sizes to build your model.
#
##### The encoders in this project utilize "separable convolution" and "batch normalization." By having separable convolution (additional layers) allow for the number of hyperparameters to be reduced at an added cost of "depth." Batch normalization allows for the minimization of the batch by adjusting for the "mean" value. This is a very common technique for neural networks in order to improve training rates and network training speed. Encoders create a set of feature maps from a filter bank by using convolutions. After batch normalization, an ReLU is applied with max pooling to acheive translation invarience. Since the boundary data must be stored, there can be issues if convolutions are called in an incorrect order. One thing to note is, by storing this data (memory) the overall accuracy decreases and the cost of the system increases. 
#
##### Decoder blocks have a different tasks. The selection for this project is to include an upsampler (bilinear) layer and a concatenation step.  The upsampler takes a "weighted average" of pixel data from the adjacent pixels to return a more useful value (average). The concatination layer chains the upsample and large input layers together. Finally, the output layer performs another separable convolution to retain the spatial information. The bilinear upsampler does interpolation in one direction and then in another (similar to a forward/backward propagation). This allows for a relatively realistic image to be produced, since the information is represented. Also, with respect to data, some information is technically "lost" since it is a weighted average. This can be problematic if there is an application (military, noise) in which the entire dataset (analog for instance) needs to be preserved.
#
##### 1x1 convolutions are useful for the reduction of dimensionality while retaining data. Essentially, the 1x1 convolution layer behaves as a linear coordinate-dependent transformation in the filter space. It should be noted that its usage is a funtion of kernel size and results in less over-fitting (especially when using stochastic gradient descent). 1x1 convolutions, while mathematically equivalent to Fully Connected Layers (FCL), are more flexible. FCLs require a fixed size, where as the 1x1 can accept various values. 1x1 convolutions help increase model accuracy while allowing new parameters and non-linearity. Since the FCL will contrain the network to its specifications, using a 1x1 is more beneficial for dimensionality reduction. 
#
##### Skip connections work to improve the gradient flow through the network. This effectively increases the capacity without increasing the number of parameters. Skip connections are not often used in networks that are smaller than ten layers, due to the benefits appearing from traversing the gradient in many layers. This is similar to taking many samples (or integral slices in calculus) to gain very small gains over a large number of processes to result in a large summation of gains.
#
###### http://iamaaditya.github.io/2016/03/one-by-one-convolution/
###### https://datascience.stackexchange.com/questions/12830/how-are-1x1-convolutions-the-same-as-a-fully-connected-layer
###### https://stackoverflow.com/questions/39366271/for-what-reason-convolution-1x1-is-used-in-deep-neural-networks
###### https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network
###### 
###### 
### Encoder Block
___


##### Create an encoder block that includes a separable convolution layer using the `separable_conv2d_batchnorm()` function. The `filters` parameter defines the size or depth of the output layer. For example, 32 or 64. The separable convolution layers allow for the reduction of the total number of parameters needed. 

```
def encoder_block(input_layer, filters, strides):

    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)

    return output_layer
```

### Decoder Block
___

##### The decoder block is comprised of three parts:

* A bilinear upsampling layer using the upsample_bilinear() function. The current recommended factor for upsampling is set to 2.
* A layer concatenation step. This step is similar to skip connections. You will concatenate the upsampled small_ip_layer and the large_ip_layer.
* Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.
#

```
def decoder_block(small_ip_layer, large_ip_layer, filters):

    # Upsample the small input layer using the bilinear_upsample() function.
    small_upsample_layer = bilinear_upsample(small_ip_layer)
    # Concatenate the upsampled and large input layers using layers.concatenate
    concatenation_layers = layers.concatenate([small_upsample_layer, large_ip_layer])
    # Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concatenation_layers, filters)
    return output_layer
```

### Model
___

##### Now that that the encoder and decoder are blocks ready, the FCN can be built.
#

##### There are three steps:

* Add encoder blocks to build the encoder layers. 
* Add a 1x1 Convolution layer using the conv2d_batchnorm() function. Remember that 1x1 Convolutions require a kernel and stride of 1.
* Add decoder blocks for the decoder layers.


```
def fcn_model(inputs, num_classes):

    # Add Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    additional_layer = conv2d_batchnorm(inputs, 16, 1, 1)
    encoding_layer1 = encoder_block(additional_layer, 32, 2)           
    encoding_layer2 = encoder_block(encoding_layer1, 64, 2)
    encoding_layer3 = encoder_block(encoding_layer2, 128, 2)

    # Add 1x1 Convolution layer using conv2d_batchnorm().
    convolution_layer = conv2d_batchnorm(encoding_layer3, 256, 1, 1)

    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoding_layer1 = decoder_block(convolution_layer, encoding_layer2, 128)     # note default kernel_size=3
    decoding_layer2 = decoder_block(decoding_layer1, encoding_layer1, 64)
    decoding_layer3 = decoder_block(decoding_layer2, additional_layer, 32)

    x = decoding_layer3

    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```

## Training 
___
##### The following cells will use the FCN created and define an ouput layer based on the size of the processed image and the number of classes recognized. Hyperparameters must be defined and implemented to train the network.
#

#####  Please Note: For this project, the helper code in `data_iterator.py` will resize the copter images to 160x160x3 to speed up training.
#
#


```
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

image_hw = 160
image_shape = (image_hw, image_hw, 3)
inputs = layers.Input(image_shape)
num_classes = 3

# Call fcn_model()
output_layer = fcn_model(inputs, num_classes)
```

### Hyperparameters
___
##### Note: The hyperparameters were chosen based on previous instruction, intuition and ultimately trial-and-error. The training rate was chosen to be slightly below .5, as it resulted in a closer fit to the model. A relatively high number (40) of epochs was chosen, to have the training set propagated more times to attempt at convergence between the training and validation sets. Likewise, a higher amount of steps (based on the number of available images) resulted in better results, as to be expected. The changes did increase the computational time (which can be mitigated in the cloud). Since there was not time requirement (let's plan for this drone to not attempt computation in real-time), it was justifiable to allow a higher time allowance for a better score.
#
###### Learning Rate - How quickly the network learns how to identify the chosen object. A higher learning rate is faster to train, at a cost of being less accurate. A smaller learning rate is more accurate as the machine does not change its mind as quickly, at a nonlinear (maybe O(2^n)) increase in time below a certain value (approaching zero).
#
###### Number of Epochs - Number of propagations. Imagine washing your clothes 20 times to get them really clean. There is a point in which they wil not become more clean, and the ideal number (amout of times) can be arrived at by trial and error or knowing the impacts of the other hyperparameters on the overall network or system.
#
###### Steps per Epoch - the number of training images sent through each epoch. Imagine washing ten socks in a load of laundry. 
#
###### Validation Steps - Like the steps per epoch, except it is specifically for the validation set.
#
###### Workers - How many processes to work on the problem. More processes is better (AWS uses 4 for Large p2 clusters) but using only 2 might be more feasible on a personal computer (note: this will take a very long time depending on epoch number and learning rate, easily more than 24 hours below .5 learning rate and more than 20 epochs).
#
###### Batch Size - Example: if you have 1000 images and a batch size of 100, the algorithm will send 100 images through the training process at a time. Larger batch sizes process more data at a time, but at a higher computational cost trade-off to decreased amount of memory required.
#
###### https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
###### https://www.quora.com/What-is-the-learning-rate-in-neural-networks
#
##### The Udacity definitions for the hyperparameters are:

* **batch_size**: number of training samples/images that get propagated through the network in a single pass.
* **num_epochs**: number of times the entire training dataset gets propagated through the network.
* **steps_per_epoch**: number of batches of training images that go through the network in 1 epoch. We have provided you with a default value. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.
* **validation_steps**: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. We have provided you with a default value for this as well.
* **workers**: maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware. We have provided a recommended value to work with.

```
learning_rate = 0.004
batch_size = 32
num_epochs = 40
steps_per_epoch = 200
validation_steps = 50
workers = 4
```

##### Define the Keras model and compile it for training
#
#

```
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

model = models.Model(inputs=inputs, outputs=output_layer)

model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

# Data iterators for loading the training and validation data
train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                               data_folder=os.path.join('..', 'data', 'train'),
                                               image_shape=image_shape,
                                               shift_aug=True)

val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                             data_folder=os.path.join('..', 'data', 'validation'),
                                             image_shape=image_shape)

logger_cb = plotting_tools.LoggerPlotter()
callbacks = [logger_cb]

model.fit_generator(train_iter,
                    steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                    epochs = num_epochs, # the number of epochs to train for,
                    validation_data = val_iter, # validation iterator
                    validation_steps = validation_steps, # the number of batches to validate on
                    callbacks=callbacks,
                    workers = workers)
```

##### The images below show the comparisions between the epoch(3) and epoch(40). It can be seen that these hyperparameters allow a relatively close following of the curves for the training and validation sets, with respect to the error loss.
#
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/000.png?raw=true?raw=true)
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/002.png?raw=true?raw=true)





#
## Prediction
#
##### Now that the model is trained and saved, it can be used to make predictions on the validation dataset. These predictions can be compared to the mask images, which are the ground truth labels, to evaluate how well your model is doing under different conditions. 
#
##### There are three different predictions available from the helper code provided:

* **patrol_with_targ**: Test how well the network can detect the hero from a distance.
* **patrol_non_targ**: Test how often the network makes a mistake and identifies the wrong person as the target.
* **following_images**: Test how well the network can identify the target while following them.
#
##### The following cell will write predictions to files and return paths to the appropriate directories.
#
##### The `run_num` parameter is used to define or group all the data for a particular model run. You can change it for different runs. For example, 'run_1', 'run_2' etc.
#
#####
#

```
run_num = 'run_3'

val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                        run_num,'patrol_with_targ', 'sample_evaluation_data')

val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model,
                                        run_num,'patrol_non_targ', 'sample_evaluation_data')

val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                        run_num,'following_images', 'sample_evaluation_data')
```

##### Compare the predictions, and compare them to the ground truth labels and original images.
##### The following cells are ran to visualize some sample images from the predictions in the validation set.
#
##### Images while following the target
#
```
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','following_images', run_num)
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
```
#
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/003.png?raw=true?raw=true)
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/004.png?raw=true?raw=true)
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/005.png?raw=true?raw=true)
#
##### Images while at patrol without target
#
```
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_non_targ', run_num)
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
```
#
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/006.png?raw=true?raw=true)
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/007.png?raw=true?raw=true)
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/008.png?raw=true?raw=true)
#
##### Images while at patrol with target
#
```
im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_with_targ', run_num)
for i in range(3):
    im_tuple = plotting_tools.load_images(im_files[i])
    plotting_tools.show_images(im_tuple)
```
#
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/009.png?raw=true?raw=true)
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/010.png?raw=true?raw=true)
![alt text](https://github.com/Ohara124c41/RoboND-Follow_Me/blob/master/images/011.png?raw=true?raw=true)
#
## Evaluation
#
#### Now, the model is evaluated using the following conditions and formulae:
#
##### Scores for while the quad is following behind the target.
#
```
true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)
```
#
```
number of validation samples intersection over the union evaulated on 542
average intersection over union for background is 0.9957179100643849
average intersection over union for other people is 0.39279620415847066
average intersection over union for the hero is 0.9063536271767668
number true positives: 539, number false positives: 0, number false negatives: 0
```
#
##### Scores for images while the quad is on patrol and the target is not visable
#
```
true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)
```
#
```
number of validation samples intersection over the union evaulated on 270
average intersection over union for background is 0.9901814911147127
average intersection over union for other people is 0.8145893794238864
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 36, number false negatives: 0
```
#
##### This score measures how well the neural network can detect the target from far away
#
```
true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)
```
#
```
number of validation samples intersection over the union evaulated on 322
average intersection over union for background is 0.9971254607964568
average intersection over union for other people is 0.5102308567179387
average intersection over union for the hero is 0.42784565743401926
number true positives: 197, number false positives: 3, number false negatives: 104
```
#
##### Sum all the true positives, etc from the three datasets to get a weight for the score
#
```
true_pos = true_pos1 + true_pos2 + true_pos3
false_pos = false_pos1 + false_pos2 + false_pos3
false_neg = false_neg1 + false_neg2 + false_neg3

weight = true_pos/(true_pos+false_neg+false_pos)
= 0.8373151308304891
```
#
##### The IoU for the dataset that never includes the hero is excluded from grading:
#
```
final_IoU = (iou1 + iou3)/2
= 0.667099642305
```
#
##### The final score is given by:
#
```
final_score = final_IoU * weight
```
* ### Final Score = 0.558572624274

#
## Discussion & Future Improvements
___
#
##### Now that the expected performance can be seen (and tested in the simulation) further applications should be considered. There is the question of "would this work for a dog, car or some other object?" The answer, like in my robotic applications is, "it depends." In order for the program and algorithms to work for another object would require a new dataset from an image stream. However, most of the core code can be used as-is or with some minor modifications. Similarly, the hyperparameter selection we used can give us some intuition when applying this information to the future object of interest.
# 
##### There are a few suggestions for improvements. The primary consideration should be given to speed of computation. While this project is very beneficial as a learning exercise and for tracking a dedicated object, it will have major issues in a real-time environment. Consider, in this project, if a new target was requested or the current target changed clothing in the street. This particular drone is "not intelligent" so it will just report that the target was lost. 
# 
##### Another issue is feasibility. To acquire nominal results, many pictures of the target were taken for training. This makes training a neural network with very limited or non-precise images unreliable. Likewise, if there is poor visibility (like fog working as a light diffuser) the results will not be nominal. This is a very common issue for military tracking applications and robotics. Other sensors need to be implemented (IR, Ultrasonic, GPS, IMU) and with a combination of onboard processing (MCUs, FPGAs) and cloud computation. This shows that there will be a strong relationship between robotics, AI, real-time OS for embedded systems, and IoT for the optimization of tasks in the future. 
# 
##### Finally, not too much time was spent in the construction of the hyperparameters. They could be derived with a more theoretical (or compared with previous work) to find the best learning rate, batch size, and step size. Collecting more images for training might have been beneficial too, up until a point (recall: trade-off between training speed and accuracy). In short, the score could be optimized greatly at the expense of a deeper investigation. 

