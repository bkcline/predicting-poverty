import os
import gdal
import pandas as pd

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
    files_df = pd.DataFrame(columns = ['filepath', 'filename'])
    for subdir, dirs, files in os.walk(rootdir):
        for cur_file in files:
            filepath =  os.path.join(subdir, cur_file)
            if filepath.endswith(filetype):
                print filepath
                files_df = files_df.append({'filepath': filepath, 'filename': cur_file},
                            ignore_index = True)
    return files_df


def get_latitude_and_longitude_from_filename(filename):
    #assumes image name format: 'image_lat_lon_...''
    #ie. image_-0.008333_31.783332_400x400_zoom16_1503677509.png
    filename_components = filename.split('_')
    latitude = float(filename_components[1])
    longitude = float(filename_components[2])
    return latitude, longitude


def get_nightlights_data(nightlights_filepath, files_df):
    ds = gdal.Open(nightlights_filepath)
    nightlights_data = ds.ReadAsArray()
    nightlights_geotransform = ds.GetGeoTransform()
    nightlights_projection = ds.GetProjection()
    print 'shape of nightlights data: ', nightlights_data.shape
    print 'geotransform: ', nightlights_geotransform
    print 'projection: ', nightlights_projection

    files_df['latitude'], files_df['longitude'] =  zip(*files_df['filename'].apply(lambda x: get_latitude_and_longitude_from_filename(x)))
    files_df['nightlights_value'] = files_df.apply(
                                lambda x: get_value_at_point(nightlights_data,
                                                            nightlights_geotransform,
                                                            x['latitude'],
                                                            x['longitude']), axis = 1)
    return files_df


'''
def get_nightlights_data(nightlights_filepath, list_of_filepaths):
    ds = gdal.Open(nightlights_filepath)
    nightlights_data = ds.ReadAsArray()
    nightlights_geotransform = ds.GetGeoTransform()
    nightlights_projection = ds.GetProjection()
    print 'shape of nightlights data: ', nightlights_data.shape
    print 'geotransform: ', nightlights_geotransform
    print 'projection: ', nightlights_projection

    list_of_nightlights_values = []
    for filepath in list_of_filepaths:
        filename = get_filename_from_filepath(filepath)
        latitude, longitude = get_latitude_and_longitude_from_filename(filename)
        nightlights_value = get_value_at_point(data, geotransform, latitude, longitude)
        list_of_nightlights_values.append(nightlights_value)
    return list_of_nightlights_values
'''


if __name__ == '__main__':
    rootdir = 'data/malawi_images'
    nightlights_filepath = '../../DMSP/F182013.v4c_web.avg_vis.tif'

    files_df= get_list_of_filepaths(rootdir)
    print files_df
    nightlights_df = get_nightlights_data(nightlights_filepath, files_df)
    print nightlights_df
