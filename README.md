# COVID-19_Detector_X-RAY
I have developed an AI model to combat and understand the coronavirus (COVID-19) which affects the whole world currently. My CNN model, which I use various datasets to train it, can detect COVID-19 with chest X-RAY pictures. I hope to develop the precise model as the data in my hand increases day by day.
![X-ray image comparison, the picture is taken from pyimagesearch](https://github.com/mcagriaksoy/COVID-19_Detector_X-RAY/blob/master/covid_comp.jpg)
Codeblock is created by me!
The dataset I choosed is related the Covid-19 and the Pneumonia detection. It consists
the x-ray images of various chests. It is more than 2GB and consists very different images.
The link: kaggle.com/paultimothymooney/chest-xray-pneumonia
I have made the dataset for tensorflow competible. My changes are seperation the data
for binary classification while seperating test,train folders.The I have defined the train
covid and normal(healthly) people folders.

datasetDir = os.path.join(my google drive path, 'dataset')
train_covid_dir = os.path.join(datasetDir, 'covid')
train_normal_dir = os.path.join(datasetDir, 'normal')

After that Keras function of “flow_from_directory” is used for creating train and the
validation data as like tf.data.Dataset from image files in a directory.
Then, for data augmentation; I’d like to see the data from the dataset. I have displayed
the data and after investigation the all Images, realized that the number of normal and
pneumonia image numbers. That is very informative for me while creating the model and
configuring it. I need to avoid the overfitting problem so that I’d like to use data
augmentation to create new images and enhance the model behavior.
Data augmentation is used with thanks the function of ImageDataGenerator.This function
takes some parameters like:
```python
image_gen_train = ImageDataGenerator(
 rotation_range=15,
 width_shift_range=0.01,
 height_shift_range=0.01,
 rescale=1./255,
 shear_range=0.1,
 fill_mode='nearest',
 validation_split=0.2)
 ```
In my example, I created the images while rotating the input data and shifting width,
height, and rescaling these. (for faster training time)

Although the ImageDataGenerator function does not simply do the additive operation as
everyone wants, it allows us to train our model with the data of the magic 😊 function
creates. We specify the images’ directories, and it generates the training and the validation
augmented data for each epoch.

For example, if we have 100 images in total, with this function we will have another 100
different transformed and augmented or we can say enhanced images per epoch.
After generating the new images for the data augmentation, we need to adapt these in
train and test pools. I used following code for the purpose:
```python
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
 directory=datasetDir,
shuffle=True,
 target_size=(IMG_HEIGHT, IMG_WIDTH),
 class_mode='binary')
val_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
 directory=datasetDir,
 shuffle=True,
 target_size=(IMG_HEIGHT, IMG_WIDTH),
 class_mode='binary')
 ```
 This flow_from_directory function is described in keras documentation as “generating
function a tf.data.dataset from image files in a dir” and it must be called just after that
the function of ImageDataGenerator.
For question 2; I have downloaded the known model MobileNetV2 for my task.
MobileNetV2 is a CNN architecture that is designed for the Image classification tasks. 

I got the model like in the code:
```python
base_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3),
include_top = False, weights = "imagenet")
```
The model named the base_model and the input shape as same as my x-ray images. Input
is RGB and the weights of the downloaded model is get from the “imagenet” competition.
After my model layer, structure defining I need to compile it as some parameters. I have
used the Adam optimizer (the learning rate is decreased 0.00001 its because to gather
smoother curve for better displaying and lower learning rate is better for this image
learning due to my previous experiments) to and BinaryCrossentropy loss method is used also, 
metrics are described as their 'accuracy'.
```python
base_learning_rate = 0.00001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
 metrics=['accuracy'])
 ```
 And the most important step is generating the model. In this step model is learning itself
from our dataset which we divided test and training. It is learning from its train folder and
test itself as we expected from the folder name “test”. It is training from the
“train_data_gen” folder, our predefined steps and epochs. Also same configs are in the
validation.
```python
history = model.fit_generator(
 train_data_gen,
 steps_per_epoch= train_data_gen.samples // batch_size,
 epochs=epochs,
 validation_data=val_data_gen,
 validation_steps= val_data_gen.samples // batch_size
)
 ```
In summary of part 2, transfer learning is a technique that is used many times in the
literature. Thanks to transfer learning the pre-trained model for a different task can be
used for different task second time.
Comparison between unaugmented and the augmented data solutions of the
MobileNetv2 model.
Simply, In second time, I have called the image ImageDataGenerator functions without
any parameter. It simply means that the func. Does not create any augmentation due to
its handbook.
![Accuracies Compared](https://github.com/mcagriaksoy/COVID-19_Detector_X-RAY/blob/master/Acc1.PNG)

# Summary
The model is detecting covid-19 from previous patients' chest x-ray images.
![X-ray image](https://github.com/mcagriaksoy/COVID-19_Detector_X-RAY/blob/master/x-ray.JPG)
# Accuracy results:
![Results](https://github.com/mcagriaksoy/COVID-19_Detector_X-RAY/blob/master/Plot.PNG)

The idea and dataset comes from:

 Adrian Rosebrock, PhD 
 
 https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/
 
 https://github.com/ieee8023/covid-chestxray-dataset
 
 https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
