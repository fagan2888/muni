import json
import numpy as np
import pandas as pd
import pandas_redshift as pr
import pickle
from os import listdir
from os.path import isfile, join
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import interp1d
import scipy.stats as st

def get_distributions(sample_flag):
    with open('credentials.json') as json_data:
        credentials = json.load(json_data)

    pr.connect_to_redshift(dbname = 'muni',
                        host = 'jonobate.c9xvjgh0xspr.us-east-1.redshift.amazonaws.com',
                        port = '5439',
                        user = credentials['user'],
                        password = credentials['password'])

    if sample_flag:
        df = pr.redshift_to_pandas("""select departure_time_hour, departure_stop_id, arrival_stop_id, shape, scale, shape*scale as mean
                                        from distributions_gamma limit 1000""")
        df.to_csv('data/distributions_gamma_sample.csv', index=False)
    else:
        df = pr.redshift_to_pandas("""select departure_time_hour, departure_stop_id, arrival_stop_id, shape, scale, shape*scale as mean
                                        from distributions_gamma""")
        df.to_csv('data/distributions_gamma.csv', index=False)
    pr.close_up_shop()
    return df

def load_gtfs_data(path='google_transit'):
    #Returns dictionary of dataframes containing GTFS data
    files = [f for f in listdir(path) if isfile(join(path, f))]
    df = {}
    for file in files:
        print("Loading {}".format(file[:-4]))
        try:
            df[file[:-4]] = pd.read_csv(path + '/' + file)
        except:
            print("{} failed to load!".format(file[:-4]))
            continue
    return df


def create_features(df, df_gtfs):
    #Convert timestamps
    df['departure_time_hour'] = pd.to_datetime(df['departure_time_hour'])

    #Re-localize time
    df['departure_time_hour'] = df['departure_time_hour'].dt.tz_localize('utc').dt.tz_convert('US/Pacific')

    #Generate local day of week and hour features
    df['dow'] = df['departure_time_hour'].dt.dayofweek
    df['hour'] = df['departure_time_hour'].dt.hour

    #Append stop metadata
    df_dep = df_gtfs['stops'].copy()
    df_dep = df_dep.add_suffix('_dep')

    df_arr = df_gtfs['stops'].copy()
    df_arr = df_arr.add_suffix('_arr')

    df = df.merge(df_dep, left_on='departure_stop_id', right_on='stop_code_dep')
    df = df.merge(df_arr, left_on='arrival_stop_id', right_on='stop_code_arr')

    #Calculate stop distances
    df['stop_lat_dist'] = (df['stop_lat_dep'] - df['stop_lat_arr'])
    df['stop_lon_dist'] = (df['stop_lon_dep'] - df['stop_lon_arr'])
    df['stop_dist'] = np.sqrt((df['stop_lat_dist']**2) + (df['stop_lon_dist']**2))

    #Drop null columns, drop string/datetime columns
    df = df.dropna(axis='columns', how='all')
    df = df.drop(df.select_dtypes(['object', 'datetime64[ns, US/Pacific]']), axis=1)

    #Drop rows with nulls
    df = df.dropna(axis='rows', how='any')

    #Reset index
    df = df.reset_index(drop=True)
    return df

def fit_default(X, y, name, sample_flag):

    clf = RandomForestRegressor(oob_score=True)

    clf.fit(X, y)

    # Save the  model to a pickle file
    if sample_flag:
        pickle.dump(clf, open('{}_sample_planB.pickle'.format(name), 'wb'))
    else:
        pickle.dump(clf, open('{}_planB.pickle'.format(name), 'wb'))

    return clf

def grid_search(X, y, name, sample_flag):

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    model = RandomForestRegressor()

    # Create the GridSearch model
    clf = RandomizedSearchCV(estimator = model,
                                param_distributions = random_grid,
                                n_iter = 10,
                                cv = 3,
                                verbose=10,
                                random_state=42,
                                n_jobs = -1)

    # Fit the GridSearch model
    clf.fit(X, y)

    # Save the best model to a pickle file
    if sample_flag:
        pickle.dump(clf.best_estimator_, open('{}_sample_planB.pickle'.format(name), 'wb'))
    else:
        pickle.dump(clf.best_estimator_, open('{}_planB.pickle'.format(name), 'wb'))

    return clf.best_estimator_
