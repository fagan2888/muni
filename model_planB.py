from importlib import reload
import numpy as np
import pandas as pd
import time
import data_processing_planB as dp
from scipy.interpolate import interp1d
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor
import pickle
import datetime
reload(dp)

start_time = time.time()

SAMPLE_ONLY = False

REFRESH_DATA = False
TRAIN_MODEL = True
TEST_MODEL = False

#Load data
if REFRESH_DATA:
    print('Refreshing raw data from SQL...')
    df = dp.get_distributions(SAMPLE_ONLY)
else:
    print('Loading raw data from CSV...')
    if SAMPLE_ONLY:
        df = pd.read_csv('data/distributions_gamma_sample.csv')
    else:
        df = pd.read_csv('data/distributions_gamma.csv')

#Load GTFS data
#df_gtfs = dp.load_gtfs_data()
#pickle.dump(df_gtfs, open('df_gtfs.pickle', 'wb'))
df_gtfs = pickle.load(open('df_gtfs.pickle', 'rb'))

if TRAIN_MODEL:
    print('Engineering features... ({} secs elapsed)'.format(time.time() - start_time))

    df = df.dropna()

    #Split into X and y
    y_mean = df['mean']
    y_shape = df['shape']
    X_mean = df.drop(columns=['mean', 'shape', 'scale'])

    #Create Features
    X_mean = dp.create_features(X_mean, df_gtfs)
    pickle.dump(X_mean, open('X_mean.pickle', 'wb'))

    #X_mean = pickle.load(open('X_mean.pickle', 'rb'))

    print('Creating models from distributions... ({} secs elapsed)'.format(time.time() - start_time))
    #Train model to predict mean
    #gs_mean, clf_mean = dp.grid_search(X_mean, y_mean, 'clf_mean', SAMPLE_ONLY, start_time)
    clf_mean = dp.fit_default(X_mean, y_mean, 'clf_mean', SAMPLE_ONLY, start_time)

    print(clf_mean)
    print(pd.DataFrame(clf_mean.feature_importances_,index = X_mean.columns,columns=['importance']).sort_values('importance',ascending=False))

    #Predict means from clf_mean model and add back into training data
    y_mean_pred = pd.DataFrame(clf_mean.predict(X_mean), columns=['mean'])
    X_shape = X_mean.merge(y_mean_pred, left_index=True, right_index=True)

    #Train model to predict shape
    #gs_shape, clf_shape = dp.grid_search(X_shape, y_shape, 'clf_shape', SAMPLE_ONLY, start_time)
    clf_shape = dp.fit_default(X_shape, y_shape, 'clf_shape', SAMPLE_ONLY, start_time)

    print(clf_shape)
    print(pd.DataFrame(clf_shape.feature_importances_,index = X_shape.columns,columns=['importance']).sort_values('importance',ascending=False))

else:
    print('Loading models from pickle files... ({} secs elapsed)'.format(time.time() - start_time))
    #Reload model so that model names are standard
    if SAMPLE_ONLY:
        clf_mean = pickle.load(open('clf_mean_sample_planB.pickle', 'rb'))
        clf_shape = pickle.load(open('clf_shape_sample_planB.pickle', 'rb'))
    else:
        clf_mean = pickle.load(open('clf_mean_planB.pickle', 'rb'))
        clf_shape = pickle.load(open('clf_shape_planB.pickle', 'rb'))

if TEST_MODEL:
    cols = ['departure_time_hour','departure_stop_id','arrival_stop_id']

    #This example is Castro/Church/VN/CC/Powell to Montgomery, all lines
    data = [
            ['2018-11-21 08:00-08:00', 15728, 15731],
            ['2018-11-21 08:00-08:00', 15726, 15731],
            ['2018-11-21 08:00-08:00', 15419, 15731],
            ['2018-11-21 08:00-08:00', 15727, 15731],
            ['2018-11-21 08:00-08:00', 15417, 15731],
            ]
    df_test = pd.DataFrame(data, columns=cols)

    X_mean = dp.create_features(df_test.copy(), df_gtfs)

    #Predict means from clf_mean model and add back into test data
    y_mean_pred = pd.DataFrame(clf_mean.predict(X_mean), columns=['mean'])
    X_shape = X_mean.merge(y_mean_pred, left_index=True, right_index=True)

    #Predict means from clf_mean model and add back into test data
    y_shape_pred = pd.DataFrame(clf_shape.predict(X_shape), columns=['shape'])

    df_test = df_test.merge(y_mean_pred, left_index=True, right_index=True)
    df_test = df_test.merge(y_shape_pred, left_index=True, right_index=True)

    print (df_test)
