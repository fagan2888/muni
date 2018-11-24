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
    df_trips = df_gtfs['trips']
    df_routes = df_gtfs['routes']
    df_stops = df_gtfs['stops']
    df_stop_times = df_gtfs['stop_times']

    #PROCESS INPUT DATA
    #Convert timestamps
    df['departure_time_hour'] = pd.to_datetime(df['departure_time_hour'])

    #Re-localize time
    df['departure_time_hour'] = df['departure_time_hour'].dt.tz_localize('utc').dt.tz_convert('US/Pacific')

    #Generate local day of week and hour features
    df['dow'] = df['departure_time_hour'].dt.dayofweek
    df['hour'] = df['departure_time_hour'].dt.hour

    #Create service ID (1=weekday, 2=saturday, 3=sunday)
    df['service_id'] = 1
    df['service_id'][df['dow'] == 5] = 2
    df['service_id'][df['dow'] == 6] = 3

    #FIND ROUTE NUMBERS
    #Merge dataframes together to departures
    df_trip_pairs = pickle.load(open('df_trip_pairs.pickle', 'rb'))
    df_trip_pairs['departure_stop_id'] = ('1' +df_trip_pairs['departure_stop_id'].astype(str)).astype(int)
    df_trip_pairs['arrival_stop_id'] = ('1' +df_trip_pairs['arrival_stop_id'].astype(str)).astype(int)
    df = df.merge(df_trip_pairs, on=['service_id', 'departure_stop_id', 'hour', 'arrival_stop_id'], how='left')

    #ADD STOP METADATA
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

    df = df.fillna(0)

    #Reset index
    df = df.reset_index(drop=True)
    return df

def fit_default(X, y, name, sample_flag):

    clf = RandomForestRegressor(oob_score=True)

    clf.fit(X, y)

    # Save the  model to a pickle file
    print('Writing {} model to pickle file...'.format(name))
    if sample_flag:
        pickle.dump(clf, open('{}_sample_planB.pickle'.format(name), 'wb'))
    else:
        pickle.dump(clf, open('{}_planB.pickle'.format(name), 'wb'))

    return clf

def grid_search(X, y, name, sample_flag):

    # Number of trees in random forest
    n_estimators = [10,20]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [10]
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

    model = RandomForestRegressor(oob_score=True)

    # Create the GridSearch model
    gs = RandomizedSearchCV(estimator = model,
                                param_distributions = random_grid,
                                n_iter = 10,
                                cv = 3,
                                verbose=10,
                                random_state=42,
                                n_jobs = -1)

    # Fit the GridSearch model
    gs.fit(X, y)

    # Save the best model to a pickle file
    if sample_flag:
        pickle.dump(gs.best_estimator_, open('{}_sample_planB.pickle'.format(name), 'wb'))
    else:
        pickle.dump(gs.best_estimator_, open('{}_planB.pickle'.format(name), 'wb'))

    return gs, clf.best_estimator_
