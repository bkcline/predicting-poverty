from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.models import Model
import numpy as np

#https://stackoverflow.com/questions/43867032/how-to-fine-tune-resnet50-in-keras

image_directory = 'dogscats'
height = 224
width = 224
color_mode = 'rgb'
shuffle = True
seed = 0
channels = 3
data_format = 'channels_last'
if data_format == 'channels_last':
    input_shape = (height, width, channels)
else:
    input_shape = (channels, height, width)
batch_size = 32
epochs = 1
n_classes = 2
lr = .0001
model_filename = 'cnn3.h5'

def preprocess_input_wrapper(x):
    """
    Source: https://nbviewer.jupyter.org/gist/embanner/6149bba89c174af3bfd69537b72bca74
    Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.

    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)

    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.

    Returns a numpy 3darray (the preprocessed image).

    """
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

train_datagen = ImageDataGenerator(
    #rescale = 1. / 255,
    preprocessing_function = preprocess_input_wrapper,
    data_format = data_format)

test_datagen = ImageDataGenerator(
    #rescale = 1. / 255,
    preprocessing_function = preprocess_input_wrapper,
    data_format = data_format)

train_generator = train_datagen.flow_from_directory(
        directory = image_directory + '/train',
        target_size = (height, width),
        color_mode = color_mode,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = seed)

validation_generator = test_datagen.flow_from_directory(
        directory = image_directory + '/valid',
        target_size = (height, width),
        color_mode = color_mode,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = seed)

training_steps_per_epoch = max(1,int(train_generator.samples / batch_size))
validation_steps_per_epoch = max(1,int(validation_generator.samples / batch_size))
print training_steps_per_epoch, train_generator.samples
print validation_steps_per_epoch, validation_generator.samples



base_model = ResNet50(weights='imagenet', include_top = False, input_shape = input_shape)
x = base_model.output
x = Flatten(name = 'output_features')(x) #using x = GlobalAveragePooling2D()(x) instead of flatten would enable different input shapes to be used - see https://stackoverflow.com/questions/43867032/how-to-fine-tune-resnet50-in-keras
predictions = Dense(n_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.summary()
model.compile(optimizer=Adam(lr = lr), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
        train_generator,
        steps_per_epoch=training_steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps_per_epoch,
        verbose = 1)

model.save(model_filename)
print 'wrote full model'
'''
model.save_weights('cnn2_take1_weights.h5')
with open("cnn2_take1.json", "w") as json_file:
    json_file.write(model.to_json())
'''
