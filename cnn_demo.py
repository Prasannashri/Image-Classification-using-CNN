#part-1 Data Preprocessing
#part-1 can be done manually in dataset because of full of images,it can be splitted as training and test set
#part -2  Building the CNN
#imports
from keras.models import Sequential #used to initialize the NN
from keras.layers import Convolution2D #step1--convolution step
from keras.layers import MaxPooling2D #step2--MaxPool step
from keras.layers import Flatten #step3--Flattening step
from keras.layers import Dense #step4 -- Full connection step

#Initialize the NN by creating classifier object
classifier = Sequential()

#Convolution Step
classifier.add(Convolution2D(32, 3, 3, input_shape =(64,64,3) ,activation='relu'))

#Pooling Step
classifier.add(MaxPooling2D(pool_size=(2,2)))

#flattening step
classifier.add(Flatten())

#full connection
classifier.add(Dense(output_dim = 128 , activation = 'relu'))
#output layer
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

#part - 3 Fit cnn to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
#making predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
#print
print(prediction)