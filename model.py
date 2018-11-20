from importlib import reload
import numpy as np
import pandas as pd
import data_processing as dp
import scipy.stats as st
reload(dp)

REFRESH_DATA = False

#Load data
if REFRESH_DATA:
    df = dp.get_raw()
    df.to_csv('data/vehicle_monitoring.csv', index=False)

else:
    df = pd.read_csv('data/vehicle_monitoring.csv')

print('Converting raw data to stops...')
df = dp.raw_to_stops(df)

print('Converting stop data to durations...')
df = dp.stops_to_durations(df)

print('Converting durations to distributions...')
df = dp.durations_to_distributions(df)


print('Creating model from distributions...')
#Drop NaNs
df = df.dropna()

#Generate Target
df['mean'] = df['shape'] * df['scale']

#Split into X and y
y_mean = df_dists['mean']
y_shape = df_dists['shape']
X_mean = df_dists.drop(columns=['mean', 'shape', 'scale', 'sse'])

#Create Features
X_mean = dp.create_features(X_mean)

clf_mean = dp.grid_search(X_mean, y_mean, 'clf_mean')

#Predict means from clf_mean model and add back into training data
y_mean_pred = pd.DataFrame(clf_mean.predict(X_mean), columns=['mean'])
X_shape = X_mean.merge(y_mean_pred, left_index=True, right_index=True)

clf_shape = dp.grid_search(X_shape, y_shape, 'clf_shape')
