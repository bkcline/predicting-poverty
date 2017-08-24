from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.models import Model
import numpy as np

#https://stackoverflow.com/questions/43867032/how-to-fine-tune-resnet50-in-keras

image_directory = 'data'
height = 224
width = 224
color_mode = 'rgb'
shuffle = True
seed = 0
training_steps_per_epoch = 1 #*****should be n_training_images / batch_size
validation_steps_per_epoch = 1 #*****should be n_validation_images / batch_size
channels = 3
data_format = 'channels_last'
if data_format == 'channels_last':
    input_shape = (height, width, channels)
else:
    input_shape = (channels, height, width)
batch_size = 32
epochs = 3
n_classes = 2
lr = .0001



train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    data_format = data_format)

test_datagen = image.ImageDataGenerator(
    rescale=1./255,
    data_format = data_format)

train_generator = train_datagen.flow_from_directory(
        directory = image_directory + '/train',
        target_size = (height, width),
        color_mode = color_mode,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = seed)

validation_generator = test_datagen.flow_from_directory(
        directory = image_directory + '/validation',
        target_size = (height, width),
        color_mode = color_mode,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = seed)




base_model = ResNet50(weights='imagenet', include_top = False, input_shape = input_shape)
x = base_model.output
x = Flatten()(x)
#x = Dense(1024, activation='relu')(x)
predictions = Dense(n_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.summary()
model.compile(optimizer=Adam(lr = lr), loss='categorical_crossentropy', metrics = metrics=['accuracy'])
model.fit_generator(
        train_generator,
        steps_per_epoch=training_steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps_per_epoch,
        verbose = 1)
