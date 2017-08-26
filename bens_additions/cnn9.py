from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
from multi_gpu import make_parallel
from timeit import default_timer as timer
import json

class Jsonable:
    #approach to saving json from https://stackoverflow.com/questions/6578986/how-to-convert-json-data-into-a-python-object
    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)

class SharedHyperparameters(Jsonable):
#hyperparameters that will be common between train and test
    def __init__(self, height = None, width = None, color_mode = None,
                 seed = None, channels = None, data_format = None,
                 input_shape = None, multi_gpu_model_filename = None, 
                 single_gpu_model_filename = None,
                 output_features_layer_name = None):
        self.height = height
        self.width = width
        self.color_mode = color_mode
        self.seed = seed
        self.channels = channels
        self.data_format = data_format
        self.input_shape = input_shape
        self.multi_gpu_model_filename = multi_gpu_model_filename
        self.single_gpu_model_filename = single_gpu_model_filename
        self.output_features_layer_name = output_features_layer_name


class TrainAndValidationHyperparameters(Jsonable):
#hyperparameters that will used for train/validation only
    def __init__(self, epochs = None, n_classes = None, lr = None,
                 n_gpus = None, batch_size = None, shuffle = None,
                 train_image_directory = None,
                 validation_image_directory = None,
                 train_steps_per_epoch = None,
                 validation_steps_per_epoch = None):
        self.epochs = epochs
        self.n_classes = n_classes
        self.lr = lr
        self.n_gpus = n_gpus
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_image_directory = train_image_directory
        self.validation_image_directory = validation_image_directory
        self.train_steps_per_epoch = train_steps_per_epoch
        self.validation_steps_per_epoch = validation_steps_per_epoch


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


def get_train_and_validation_generators(shared_hyperparameters, train_hyperparameters):
    #function also modifies steps per epoch in train_hyperparameters

    train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input_wrapper,
        data_format = shared_hyperparameters.data_format)

    validation_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input_wrapper,
        data_format = shared_hyperparameters.data_format)

    train_generator = train_datagen.flow_from_directory(
            directory = train_hyperparameters.train_image_directory,
            target_size = (shared_hyperparameters.height, shared_hyperparameters.width),
            color_mode = shared_hyperparameters.color_mode,
            batch_size = train_hyperparameters.batch_size,
            shuffle = train_hyperparameters.shuffle,
            seed = shared_hyperparameters.seed)

    validation_generator = validation_datagen.flow_from_directory(
            directory = train_hyperparameters.validation_image_directory,
            target_size = (shared_hyperparameters.height, shared_hyperparameters.width),
            color_mode = shared_hyperparameters.color_mode,
            batch_size = train_hyperparameters.batch_size,
            shuffle = train_hyperparameters.shuffle,
            seed = shared_hyperparameters.seed)

    train_hyperparameters.train_steps_per_epoch = max(1,int(train_generator.samples / train_hyperparameters.batch_size))
    train_hyperparameters.validation_steps_per_epoch =  max(1,int(validation_generator.samples / train_hyperparameters.batch_size))
    print train_hyperparameters.train_steps_per_epoch, train_generator.samples
    print train_hyperparameters.validation_steps_per_epoch, validation_generator.samples

    return train_generator, validation_generator, shared_hyperparameters, train_hyperparameters


def build_single_gpu_model(shared_hyperparameters):
    #base_model is a pretrained imagenet cnn
    #model is the modified base_model, with different top layers
    base_model = ResNet50(weights='imagenet', include_top = False, input_shape = shared_hyperparameters.input_shape)
    x = base_model.output
    x = Flatten(name = shared_hyperparameters.output_features_layer_name)(x) #using x = GlobalAveragePooling2D()(x) instead of flatten would enable different input shapes to be used - see https://stackoverflow.com/questions/43867032/how-to-fine-tune-resnet50-in-keras
    predictions = Dense(train_hyperparameters.n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return base_model, model


def prepare_model_for_fitting_and_fit_model(base_model, model, train_hyperparameters):
    #preparation includes parallelizing and freezing layers
    '''
    Returns
    -------
    model: model that has been modified to be parallelized and has been compiled
    '''

    #first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    model.summary()
    model = make_parallel(model, train_hyperparameters.n_gpus)
    model.summary()
    model.compile(optimizer=Adam(lr = train_hyperparameters.lr), loss='categorical_crossentropy', metrics=['accuracy'])
    start = timer()
    model.fit_generator(
            train_generator,
            steps_per_epoch=train_hyperparameters.train_steps_per_epoch,
            epochs=train_hyperparameters.epochs,
            validation_data=validation_generator,
            validation_steps=train_hyperparameters.validation_steps_per_epoch,
            verbose = 1)
    end = timer()
    print(end - start)
    return model


def save_model_and_hyperparameters(multi_gpu_model,
                                shared_hyperparameters,
                                shared_hyperparameters_filename,
                                train_hyperparameters,
                                train_hyperparameters_filename):

    multi_gpu_model.save(shared_hyperparameters.multi_gpu_model_filename)
    print 'saved multi_gpu_model'

    #there's a bug in loading saved multi_gpu models
    #instead, can just save the single gpu model (since the weights are shared between the single gpu and the multi_gpu models)
    #see aschampion's comments on https://github.com/kuza55/keras-extras/issues/3
    single_gpu_model = multi_gpu_model.layers[-2]
    #compiling so it's ready to use after loading
    single_gpu_model.compile(optimizer=Adam(lr = train_hyperparameters.lr), loss='categorical_crossentropy', metrics=['accuracy'])
    single_gpu_model.save(shared_hyperparameters.single_gpu_model_filename)
    print 'saved single_gpu_model'

    with open(shared_hyperparameters_filename, 'w') as outfile:
        outfile.write(shared_hyperparameters.to_json())
    with open(train_hyperparameters_filename, 'w') as outfile:
        outfile.write(train_hyperparameters.to_json())

if __name__ == '__main__':
    shared_hyperparameters = SharedHyperparameters()
    train_hyperparameters = TrainAndValidationHyperparameters()

    #these hyperparameters are less likely to vary
    shared_hyperparameters.height = 224
    shared_hyperparameters.width = 224
    shared_hyperparameters.color_mode = 'rgb'
    shared_hyperparameters.channels = 3
    shared_hyperparameters.data_format = 'channels_last'
    shared_hyperparameters.output_features_layer_name = 'output_features'
    if shared_hyperparameters.data_format == 'channels_last':
        shared_hyperparameters.input_shape = (shared_hyperparameters.height, shared_hyperparameters.width, shared_hyperparameters.channels)
    else:
        shared_hyperparameters.input_shape = (shared_hyperparameters.channels, shared_hyperparameters.height, shared_hyperparameters.width)
    train_hyperparameters.shuffle = True

    #these hyperparemeters are more likely to vary
    shared_hyperparameters.multi_gpu_model_filename = 'cnn8_multi_gpu.h5'
    shared_hyperparameters.single_gpu_model_filename = 'cnn8_single_gpu.h5'
    shared_hyperparameters_filename = 'cnn8_shared_hyperparameters.json'
    train_hyperparameters_filename = 'cnn8_train_hyperparameters.json'
    train_hyperparameters.epochs = 5
    train_hyperparameters.n_classes = 2
    train_hyperparameters.lr = .001
    train_hyperparameters.n_gpus = 4
    train_hyperparameters.batch_size = 32* train_hyperparameters.n_gpus
    train_hyperparameters.train_image_directory = 'dogscats/train'
    train_hyperparameters.validation_image_directory = 'dogscats/valid'
    shared_hyperparameters.seed = 0

    train_generator, validation_generator, shared_hyperparameters, train_hyperparameters = get_train_and_validation_generators(shared_hyperparameters, train_hyperparameters)
    base_model, model = build_single_gpu_model(shared_hyperparameters)
    multi_gpu_model = prepare_model_for_fitting_and_fit_model(base_model, model, train_hyperparameters)
    save_model_and_hyperparameters(multi_gpu_model,
                                    shared_hyperparameters,
                                    shared_hyperparameters_filename,
                                    train_hyperparameters,
                                    train_hyperparameters_filename)
