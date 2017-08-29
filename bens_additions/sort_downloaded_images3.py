import os
import gdal
import pandas as pd
import numpy as np
import shutil

def get_value_at_point(data, geotransform, latitude, longitude):
    """Finds value of raster data at a lat and long coordinate pair.
    Implementation based on: https://waterprogramming.wordpress.com/2014/10/07/python-extract-raster-data-value-at-a-point/

    Parameters
    ----------
    data: 2D numpy array
    geotransform: format (top left x, delta x, x rotation, top left y, y rotation, delta y)
    """
    top_left_x, delta_x, x_rotation, top_left_y, y_rotation, delta_y = geotransform
    col = int((longitude - top_left_x) / delta_x)
    row = int((latitude - top_left_y) / delta_y)
    assert ((row >= 0) & (col >= 0) & (row < data.shape[0])
            & (col < data.shape[1])), "error in lookup based on given lat/long"
    return data[row, col]


def get_list_of_filepaths(rootdir, filetype = '.png'):
#this will iterate over all descendant file
# from https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
    list_of_filepaths = []
    list_of_filenames = []
    for subdir, dirs, files in os.walk(rootdir):
        for cur_file in files:
            filepath =  os.path.join(subdir, cur_file)
            if filepath.endswith(filetype):
                list_of_filepaths.append(filepath)
                list_of_filenames.append(cur_file)
    #appending to lists instead of to df directly inside loop for improved speed
    files_df = pd.DataFrame({'filepath': list_of_filepaths,
                             'filename': list_of_filenames})
    print 'completed list of filepaths'
    return files_df


def get_latitude_and_longitude_from_filename(filename):
    #assumes image name format: 'image_lat_lon_...''
    #ie. image_-0.008333_31.783332_400x400_zoom16_1503677509.png
    filename_components = filename.split('_')
    latitude = float(filename_components[1])
    longitude = float(filename_components[2])
    return latitude, longitude


def get_nightlights_bin(nightlights_value, array_of_bins, out_of_range_value = -1):
    #nightlights_bin is the arg of the corresponding bin in array_of_bins

    binned_array = np.histogram(nightlights_value, array_of_bins)
    if np.argwhere(binned_array[0] == True).size == 0:
        nightlights_bin = out_of_range_value
    else:
        nightlights_bin = np.argwhere(binned_array[0] == True).reshape(-1)[0]
    return nightlights_bin


def get_nightlights_data(nightlights_filepath, files_df, nightlights_array_of_bins):
    ds = gdal.Open(nightlights_filepath)
    nightlights_data = ds.ReadAsArray()
    nightlights_geotransform = ds.GetGeoTransform()
    nightlights_projection = ds.GetProjection()
    print 'shape of nightlights data: ', nightlights_data.shape
    print 'geotransform: ', nightlights_geotransform
    print 'projection: ', nightlights_projection
    files_df['latitude'], files_df['longitude'] =  zip(*files_df['filename'].apply(lambda x: get_latitude_and_longitude_from_filename(x)))
    print 'added lat and lon'
    files_df['nightlights_value'] = files_df.apply(
                                lambda x: get_value_at_point(nightlights_data,
                                                            nightlights_geotransform,
                                                            x['latitude'],
                                                            x['longitude']), axis = 1)
    print 'added nightlights value'
    files_df['nightlights_bin'] = files_df['nightlights_value'].apply(lambda x: get_nightlights_bin(x, nightlights_array_of_bins))
    return files_df


def select_train_and_validation_indexes(all_indices, n_train, n_validation):
    #n_train is max desired train indices
    #n_validation: should aim to get this many
    all_indices = all_indices.reshape(-1)
    print all_indices.size, n_validation
    assert all_indices.size > n_validation, 'array has too few elements'
    validation_indices = np.random.choice(all_indices, size = n_validation, replace = False)
    indices_not_for_validation = np.setdiff1d(all_indices, validation_indices)
    if n_train > indices_not_for_validation.size:
        n_train = indices_not_for_validation.size
    train_indices = np.random.choice(indices_not_for_validation, size = n_train, replace = False)
    return train_indices, validation_indices


def select_images_for_each_folder(df, nightlights_array_of_bins, n_train, n_validation):
    list_of_assignment_cols = []
    for bin_value in df.nightlights_bin.unique(): #range(nightlights_array_of_bins.size-1):
        indices = df[df['nightlights_bin']==bin_value].index.values
        print indices.shape
        train_indices, validation_indices = select_train_and_validation_indexes(indices, n_train, n_validation)
        print train_indices.shape, validation_indices.shape
        train_col = str(bin_value)+('_train')
        validation_col = str(bin_value)+('_validation')
        df[train_col] = df.index.isin(train_indices)
        df[validation_col] = df.index.isin(validation_indices)
        list_of_assignment_cols.append(train_col)
        list_of_assignment_cols.append(validation_col)
    assert df[list_of_assignment_cols].sum(axis=1).max() <= 1, 'at least one file is assigned to multiple folders'
    return df, list_of_assignment_cols


def copy_images_to_binned_folders(df, path_to_image_outputs, list_of_assignment_cols):
    if not os.path.exists(path_to_image_outputs):
        os.makedirs(path_to_image_outputs)
    for bin_column in list_of_assignment_cols:
        filepath = os.path.join(path_to_image_outputs, bin_column)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        df['filepath'][df[bin_column]==True].apply(lambda x: shutil.copy2(x, filepath))
        print 'finished copying images to folder: ', bin_column


if __name__ == '__main__':
    image_input_directory_filepath = '../data/images'
    nightlights_filepath = '../data/nightlights/F182013.v4c_web.avg_vis.tif'
    nightlights_array_of_bins = np.array([0,3,35,63]) #based on supplemental methods of jean et al 2016
    nightlights_df_filepath = '../data/output/nightlights_lsms_df.csv'
    flag_create_initial_df_with_nightlights_values = False
    flag_copy_images_to_binned_folders = True
    n_train = 50000
    n_validation = 1000
    path_to_image_outputs = '../data/output/binned_folders'

    if flag_create_initial_df_with_nightlights_values:
        files_df= get_list_of_filepaths(image_input_directory_filepath)
        nightlights_df = get_nightlights_data(nightlights_filepath, files_df, nightlights_array_of_bins)
        nightlights_df.to_csv(nightlights_df_filepath)
        print np.histogram(nightlights_df['nightlights_bin'].values, range(-1, nightlights_array_of_bins.size+1))

    if flag_copy_images_to_binned_folders:
        df = pd.read_csv(nightlights_df_filepath)
        df, list_of_assignment_cols = select_images_for_each_folder(df, nightlights_array_of_bins, n_train, n_validation)
        copy_images_to_binned_folders(df, path_to_image_outputs, list_of_assignment_cols)
