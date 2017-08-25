from keras.models import load_model
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
import numpy as np

image_directory = 'dogscats'
height = 224
width = 224
color_mode = 'rgb'
validation_shuffle = False
#when shuffle is false, should be able to map between generator.filesnames and predictions... see https://github.com/fchollet/keras/issues/6234, https://stackoverflow.com/questions/38538988/in-what-order-does-flow-from-directory-function-in-keras-takes-the-samples
#however, the alignment apparently exists only for the first time using the generator... see https://github.com/fchollet/keras/issues/3296 and brad0taylor and futurely's comments
#see also https://github.com/fchollet/keras/issues/4225#issue-186069901
#https://stackoverflow.com/questions/37981913/keras-how-to-predict-classes-in-order

######possibly use sequences instead to load data - see abnera's comment https://github.com/fchollet/keras/issues/5558
###another alternative: predict on each spererately (no generator), or load all into memory then use predict (no generator)
#see patyork's comment https://github.com/fchollet/keras/issues/5048

###WARMING: generator will not preserve order if multithreading was used: https://github.com/fchollet/keras/issues/5048
#people have tried to fix this but still dangerous: https://github.com/fchollet/keras/pull/7118

seed = 0
channels = 3
data_format = 'channels_last'
if data_format == 'channels_last':
    input_shape = (height, width, channels)
else:
    input_shape = (channels, height, width)
batch_size = 1 #this will enable predicting on all the files, guaranteed. This will be slower than a larger batch but ensures that each file is predicted on once and only once.
model_filename = 'cnn3.h5'
layer_name = 'output_features'


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

validation_datagen = ImageDataGenerator(
    #rescale = 1. / 255,
    preprocessing_function = preprocess_input_wrapper,
    data_format = data_format)

validation_generator = validation_datagen.flow_from_directory(
        directory = image_directory + '/valid',
        target_size = (height, width),
        color_mode = color_mode,
        batch_size = batch_size,
        shuffle = validation_shuffle,
        seed = seed)

list_of_filenames = validation_generator.filenames
validation_steps_per_epoch = len(list_of_filenames) 




model = load_model(model_filename)
#print model.evaluate_generator(validation_generator, validation_steps_per_epoch, max_queue_size=10, workers=1, use_multiprocessing=False)

predictions = model.predict_generator(validation_generator, validation_steps_per_epoch, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

print predictions.shape
print predictions[:20, :]
print predictions[-20:,:]
print validation_generator.classes[:20]
print list_of_filenames[:20]
print list_of_filenames[-20:]

predictions = model.predict_generator(validation_generator, validation_steps_per_epoch, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

print predictions.shape
print predictions[:20, :]
print predictions[-20:,:]

feature_extraction_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
feature_extraction_output = feature_extraction_model.predict_generator(validation_generator, validation_steps_per_epoch, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

print feature_extraction_output.shape

print validation_generator.classes[:20]
print list_of_filenames[:20]

