from importlib import reload
import numpy as np
import pandas as pd
import time
import data_processing as dp
from scipy.interpolate import interp1d
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor
import pickle
import groupby
reload(dp)

start_time = time.time()

SAMPLE_ONLY = False

REFRESH_DATA = True
PROCESS_DATA = True
TRAIN_MODEL = True
TEST_MODEL = True

#Load data
if REFRESH_DATA:
    print('Refreshing raw data from SQL...')
    df = dp.get_raw(SAMPLE_ONLY)
else:
    print('Loading raw data from CSV...')
    if SAMPLE_ONLY:
        df = pd.read_csv('data/vehicle_monitoring_sample.csv')
    else:
        df = pd.read_csv('data/vehicle_monitoring.csv')

#Load GTFS data
df_gtfs = dp.load_gtfs_data()

#Process data (or skip)
if PROCESS_DATA:
    print('Converting raw data to stops... ({} secs elapsed)'.format(time.time() - start_time))
    df = dp.raw_to_stops(df, df_gtfs)

    print('Converting stop data to durations... ({} secs elapsed)'.format(time.time() - start_time))
    df = dp.stops_to_durations(df)

    print('Converting durations to distributions... ({} secs elapsed)'.format(time.time() - start_time))
    df = dp.durations_to_distributions(df)

    if SAMPLE_ONLY:
        df.to_csv('data/distributions_sample.csv', index=False)
    else:
        df.to_csv('data/distributions.csv', index=False)
else:
    print('Loading distribution data from CSV... ({} secs elapsed)'.format(time.time() - start_time))
    if SAMPLE_ONLY:
        df = pd.read_csv('data/distributions_sample.csv', parse_dates=[0], infer_datetime_format=True)
    else:
        df = pd.read_csv('data/distributions.csv', parse_dates=[0],infer_datetime_format=True)

if TRAIN_MODEL:
    print('Creating models from distributions... ({} secs elapsed)'.format(time.time() - start_time))

    #Split into X and y
    y_mean = df['mean']
    y_shape = df['shape']
    X_mean = df.drop(columns=['mean', 'shape', 'scale'])

    #Create Features
    X_mean = dp.create_features(X_mean, df_gtfs)

    #Train model to predict mean
    clf_mean = dp.grid_search(X_mean, y_mean, 'clf_mean', SAMPLE_ONLY)
    #clf_mean = dp.fit_default(X_mean, y_mean, 'clf_mean', SAMPLE_ONLY)

    #Predict means from clf_mean model and add back into training data
    y_mean_pred = pd.DataFrame(clf_mean.predict(X_mean), columns=['mean'])
    X_shape = X_mean.merge(y_mean_pred, left_index=True, right_index=True)

    #Train model to predict shape
    clf_shape = dp.grid_search(X_shape, y_shape, 'clf_shape', SAMPLE_ONLY)
    #clf_shape = dp.fit_default(X_shape, y_shape, 'clf_shape', SAMPLE_ONLY)

else:
    print('Loading models from pickle files... ({} secs elapsed)'.format(time.time() - start_time))
    #Reload model so that model names are standard
    if SAMPLE_ONLY:
        clf_mean = pickle.load(open('clf_mean_sample.pickle', 'rb'))
        clf_shape = pickle.load(open('clf_shape_sample.pickle', 'rb'))
    else:
        clf_mean = pickle.load(open('clf_mean.pickle', 'rb'))
        clf_shape = pickle.load(open('clf_shape.pickle', 'rb'))

if TEST_MODEL:
    cols = ['departure_time_hour','route_short_name','departure_stop_id','arrival_stop_id']

    #This example is Castro to Montgomery, all lines
    data = [['2018-11-21 08:00-08:00', 'Muni-KT', 5728, 5731],
            ['2018-11-21 08:00-08:00', 'Muni-L', 5728, 5731],
            ['2018-11-21 08:00-08:00', 'Muni-M', 5728, 5731]]

    df_test = pd.DataFrame(data, columns=cols)

    X_mean = dp.create_features(df_test, df_gtfs)

    #Predict means from clf_mean model and add back into test data
    y_mean_pred = pd.DataFrame(clf_mean.predict(X_mean), columns=['mean'])
    X_shape = X_mean.merge(y_mean_pred, left_index=True, right_index=True)

    #Predict means from clf_mean model and add back into test data
    y_shape_pred = pd.DataFrame(clf_shape.predict(X_shape), columns=['shape'])

    df_test = df_test.merge(y_mean_pred, left_index=True, right_index=True)
    df_test = df_test.merge(y_shape_pred, left_index=True, right_index=True)
    print (df_test)
