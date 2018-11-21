from importlib import reload
import numpy as np
import pandas as pd
import time
import data_processing as dp
from scipy.interpolate import interp1d
import scipy.stats as st
reload(dp)

start_time = time.time()

SAMPLE_ONLY = True

REFRESH_DATA = False
PROCESS_DATA = True
TRAIN_MODEL = False

#Load data
if REFRESH_DATA:
    print('Refreshing data from SQL... ')
    df = dp.get_raw(SAMPLE_ONLY)
else:
    print('Loading data from CSV... ')
    if SAMPLE_ONLY:
        df = pd.read_csv('data/vehicle_monitoring_sample.csv')
    else:
        df = pd.read_csv('data/vehicle_monitoring.csv')

#Process data (or skip)
if PROCESS_DATA:
    print('Converting raw data to stops... ({} secs elapsed)'.format(time.time() - start_time))
    df = dp.raw_to_stops(df)

    print('Converting stop data to durations... ({} secs elapsed)'.format(time.time() - start_time))
    df = dp.stops_to_durations(df)

    print('Converting durations to distributions... ({} secs elapsed)'.format(time.time() - start_time))
    df = dp.durations_to_distributions(df)

    if SAMPLE_ONLY:
        df.to_csv('data/distributions_sample.csv', index=False)
    else:
        df.to_csv('data/distributions.csv', index=False)
else:
    if SAMPLE_ONLY:
        df = pd.read_csv('data/distributions_sample.csv')
    else:
        df = pd.read_csv('data/distributions.csv')

if TRAIN_MODEL:
    print('Creating model from distributions... ({} secs elapsed)'.format(time.time() - start_time))

    #Split into X and y
    y_mean = df_dists['mean']
    y_shape = df_dists['shape']
    X_mean = df_dists.drop(columns=['mean', 'shape', 'scale', 'sse'])

    #Create Features
    X_mean = dp.create_features(X_mean)

    clf_mean = dp.grid_search(X_mean, y_mean, 'clf_mean', SAMPLE_ONLY)

    #Predict means from clf_mean model and add back into training data
    y_mean_pred = pd.DataFrame(clf_mean.predict(X_mean), columns=['mean'])
    X_shape = X_mean.merge(y_mean_pred, left_index=True, right_index=True)

    clf_shape = dp.grid_search(X_shape, y_shape, 'clf_shape')
