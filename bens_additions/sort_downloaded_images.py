import os
import gdal

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

def get_list_of_filepaths_and_nightlights_values(rootdir, filetype = '.png'):
#this will iterate over all descendant file
# from https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
    list_of_nightlights_values = []
    list_of_filepaths = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath =  os.path.join(subdir, file)
            if filepath.endswith(filetype):
                print filepath
                latitude, longitude = get_latitude_and_longitude_from_filename(test_filename)
                nightlights_value, bin_number = get_nightlights_value(latitude, longitude)
                list_of_nightlights_values.append(nightlights_value)
                list_of_filepaths.append(filepath)
    return list_of_filepaths, list_of_nightlights_values


def get_nightlights_data(nightlights_filepath):
    ds = gdal.Open(nightlights_filepath)
    nightlights_data = ds.ReadAsArray()


if __name__ == '__main__':
    rootdir = 'AfricanCountries/'
        nightlights_filepath = 'DMSP/F182013.v4c_web.avg_vis.tif'

    list_of_filepaths, list_of_nightlights_values = get_list_of_filepaths_and_nightlights_values(rootdir)
