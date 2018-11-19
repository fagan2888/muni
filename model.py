import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import data_processing
import json
import pandas_redshift as pr

REFRESH_DATA = False

#Load data
if REFRESH_DATA:
    df_dists = data_processing.get_distributions()
    df_dists.to_csv('data/distributions_gamma.csv', index=False)

else:
    df_dists = pd.read_csv('data/distributions_gamma.csv')
    df_dists['data_frame_ref'] = pd.to_datetime(df_dists['data_frame_ref'])
    df_dists['departure_time_hour'] = pd.to_datetime(df_dists['departure_time_hour'])
    df_dists['local_departure_time_hour'] = pd.to_datetime(df_dists['local_departure_time_hour'])

#Drop NaNs
df_dists = df_dists.dropna()

#Generate Target
df_dists['mean'] = df_dists['shape'] * df_dists['scale']

#Split into X and y
y_mean = df_dists['mean']
y_shape = df_dists['shape']
X_mean = df_dists.drop(columns=['mean', 'shape', 'scale', 'sse'])

#Create Features
X_mean = data_processing.create_features(X_mean)

clf_mean = data_processing.grid_search(X_mean, y_mean, 'clf_mean')

#Predict means from clf_mean model and add back into training data
y_mean_pred = pd.DataFrame(clf_mean.predict(X_mean), columns=['mean'])
X_shape = X_mean.merge(y_mean_pred, left_index=True, right_index=True)

clf_shape = data_processing.grid_search(X_shape, y_shape, 'clf_shape')
