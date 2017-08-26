from keras.models import load_model
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
import numpy as np
from cnn8 import preprocess_input_wrapper, Jsonable, SharedHyperparameters
import json


class TestHyperparameters(Jsonable):
#hyperparameters that will used for train/validation only
    def __init__(self, batch_size = None,
                 shuffle = None,
                 image_directory = None,
                 steps_per_epoch = None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_directory = image_directory
        self.steps_per_epoch = steps_per_epoch


def get_test_generator(shared_hyperparameters, test_hyperparameters):
    #also updated test_hyperparameters

    test_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input_wrapper,
        data_format = shared_hyperparameters.data_format)

    test_generator = test_datagen.flow_from_directory(
            directory = test_hyperparameters.image_directory,
            target_size = (shared_hyperparameters.height, shared_hyperparameters.width),
            color_mode = shared_hyperparameters.color_mode,
            batch_size = test_hyperparameters.batch_size,
            shuffle = test_hyperparameters.shuffle,
            seed = shared_hyperparameters.seed)

    list_of_filenames = test_generator.filenames
    test_hyperparameters.steps_per_epoch = len(list_of_filenames)
    print 'steps per epoch: ', test_hyperparameters.steps_per_epoch
    return test_generator, test_hyperparameters, list_of_filenames


def extract_features(test_generator, test_hyperparameters, shared_hyperparameters):
    #note: using the single gpu model to avoid any potential complications regarding fileorder from using multiple gpus
    model = load_model(shared_hyperparameters.single_gpu_model_filename)
    model.summary()
    feature_extraction_model = Model(inputs=model.input,
                                     outputs=model.get_layer(shared_hyperparameters.output_features_layer_name).output)
    feature_extraction_output = feature_extraction_model.predict_generator(
                    test_generator,
                    test_hyperparameters.steps_per_epoch,
                    workers=1,
                    use_multiprocessing=False,
                    verbose=1)
    return feature_extraction_output


def save_features(feature_extraction_output,
                  feature_extraction_output_filename,
                  list_of_filenames,
                  filename_for_saving_list_of_filenames):
    print 'shape of extracted features: ', feature_extraction_output.shape
    print 'length of filename list: ', len(list_of_filenames)
    print 'initial filenames: ', list_of_filenames[:10]
    np.save(feature_extraction_output_filename, feature_extraction_output)
    np.save(filename_for_saving_list_of_filenames, np.array(list_of_filenames))
    print 'features and filenames saved'


if __name__ == '__main__':
    #when shuffle is false, should be able to map between generator.filesnames and predictions... see https://github.com/fchollet/keras/issues/6234, https://stackoverflow.com/questions/38538988/in-what-order-does-flow-from-directory-function-in-keras-takes-the-samples
    #however, the alignment apparently exists only for the first time using the generator... see https://github.com/fchollet/keras/issues/3296 and brad0taylor and futurely's comments
    #see also https://github.com/fchollet/keras/issues/4225#issue-186069901
    #https://stackoverflow.com/questions/37981913/keras-how-to-predict-classes-in-order

    ######possibly use sequences instead to load data - see abnera's comment https://github.com/fchollet/keras/issues/5558
    ###another alternative: predict on each spererately (no generator), or load all into memory then use predict (no generator)
    #see patyork's comment https://github.com/fchollet/keras/issues/5048

    ###WARMING: generator will not preserve order if multithreading was used: https://github.com/fchollet/keras/issues/5048
    #people have tried to fix this but still dangerous: https://github.com/fchollet/keras/pull/7118
    test_hyperparameters = TestHyperparameters()
    test_hyperparameters.shuffle = False #must be false for filenames to align
    test_hyperparameters.batch_size = 1 #this will enable predicting on all the files, guaranteed. This will be slower than a larger batch but ensures that each file is predicted on once and only once.

    #these are more likely to vary
    shared_hyperparameters_filename = 'cnn8_shared_hyperparameters.json'
    feature_extraction_output_filename = 'cnn8_extracted_features.npy'
    filename_for_saving_list_of_filenames = 'cnn8_extracted_features_filenames.npy'
    test_hyperparameters.image_directory = 'dogscats/valid'

    #load shared hyperparameters
    with open(shared_hyperparameters_filename, 'r') as f:
        shared_hyperparameters_json = f.read()
    shared_hyperparameters = SharedHyperparameters.from_json(shared_hyperparameters_json)

    test_generator, test_hyperparameters, list_of_filenames = get_test_generator(shared_hyperparameters, test_hyperparameters)
    feature_extraction_output = extract_features(test_generator, test_hyperparameters, shared_hyperparameters)
    save_features(feature_extraction_output,
                  feature_extraction_output_filename,
                  list_of_filenames,
                  filename_for_saving_list_of_filenames)
