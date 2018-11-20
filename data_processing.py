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

def get_raw():
    with open('credentials.json') as json_data:
        credentials = json.load(json_data)

    pr.connect_to_redshift(dbname = 'muni',
                        host = 'jonobate.c9xvjgh0xspr.us-east-1.redshift.amazonaws.com',
                        port = '5439',
                        user = credentials['user'],
                        password = credentials['password'])

    df = pr.redshift_to_pandas("""select * from vehicle_monitoring""")
    pr.close_up_shop()
    return df

def get_distributions():
    with open('credentials.json') as json_data:
        credentials = json.load(json_data)

    pr.connect_to_redshift(dbname = 'muni',
                        host = 'jonobate.c9xvjgh0xspr.us-east-1.redshift.amazonaws.com',
                        port = '5439',
                        user = credentials['user'],
                        password = credentials['password'])

    df = pr.redshift_to_pandas("""select *,
                                    convert_timezone('US/Pacific', departure_time_hour) as local_departure_time_hour
                                     from distributions_gamma""")
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


def create_features(df_dists):

    #Load GTFS data
    df_gtfs = load_gtfs_data()

    #Generate local day of week and hour features
    df_dists['local_dow'] = df_dists['local_departure_time_hour'].dt.dayofweek
    df_dists['local_hour'] = df_dists['local_departure_time_hour'].dt.hour

    #Append stop metadata
    df_dep = df_gtfs['stops'].copy()
    df_dep = df_dep.add_suffix('_dep')

    df_arr = df_gtfs['stops'].copy()
    df_arr = df_arr.add_suffix('_arr')

    df_dists = df_dists.merge(df_dep, left_on='departure_stop_id', right_on='stop_code_dep')
    df_dists = df_dists.merge(df_arr, left_on='arrival_stop_id', right_on='stop_code_arr')

    #Drop null columns, drop string/datetime columns
    df_dists = df_dists.dropna(axis='columns', how='all')
    df_dists = df_dists.drop(df_dists.select_dtypes(['object', 'datetime64']), axis=1)

    #Calculate stop distances
    df_dists['stop_lat_dist'] = (df_dists['stop_lat_dep'] - df_dists['stop_lat_arr'])
    df_dists['stop_lon_dist'] = (df_dists['stop_lon_dep'] - df_dists['stop_lon_arr'])
    df_dists['stop_dist'] = np.sqrt((df_dists['stop_lat_dist']**2) + (df_dists['stop_lon_dist']**2))

    #Reset index
    df_dists = df_dists.reset_index(drop=True)
    return df_dists


def grid_search(X, y, name):
    '''
    # Params to pass to the GridSearchCV
    param_grid = {
        'n_estimators': [10, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [None,4,6,8],
        'criterion' :['mse', 'mae']
    }
    '''

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

    print(clf.best_params_)
    print()
    print("{} OOB score: {}".format(name, clf.best_estimator_.oob_score_))
    print()
    print(pd.DataFrame(clf.feature_importances_,
            index = X.columns,
            columns=['importance']).sort_values('importance',
                                                ascending=False))
    print()

    # Save the best model to a pickle file
    pickle.dump(clf.best_estimator_, open('{}.pickle'.format(name), 'wb'))

    return clf


def raw_to_stops(df):
    #Convert datetimes
    df['recorded_time'] = pd.to_datetime(df['recorded_time'])
    df['valid_until_time'] = pd.to_datetime(df['valid_until_time'])
    df['data_frame_ref'] = pd.to_datetime(df['data_frame_ref'])
    df['expected_arrival_time'] = pd.to_datetime(df['expected_arrival_time'])
    df['expected_departure_time'] = pd.to_datetime(df['expected_departure_time'])

    #Sort values, reset index
    df = df.sort_values(['data_frame_ref', 'journey_ref', 'recorded_time'])
    df = df.reset_index(drop=True)
    df['join_index'] = df.index.astype(int)

    #Create offset dataframe with next poll data
    df_next = df[['data_frame_ref', 'journey_ref', 'recorded_time', 'stop_point_ref', 'stop_point_name']]
    df_next = df_next.add_suffix('_next')
    df_next['join_index'] = df_next.index
    df_next['join_index'] = df_next['join_index'].astype(int) - 1

    #Join data to offset data
    df = df.merge(df_next, on='join_index')

    #Filter to stop events
    df = df[(df['data_frame_ref']==df['data_frame_ref_next'])
          & (df['journey_ref']==df['journey_ref_next'])
          & (df['stop_point_ref']!=df['stop_point_ref_next'])]

    #Add in stop time column
    df['stop_time'] = df['recorded_time'] + (df['recorded_time_next'] - df['recorded_time']) / 2

    #Drop uneeded columns
    df = df[['data_frame_ref', 'journey_ref', 'stop_point_ref', 'stop_time']]

    #Rename columns to match stop data.
    df = df.rename(index=str, columns={"journey_ref": "trip_id", "stop_point_ref": "stop_id"})

    #Load stop data
    df_stop_times = pd.read_csv('google_transit/stop_times.txt')

    #Fix to deal with the fact that that stop_ids are in a slightly different format
    df_stop_times['stop_id'] = ('1' + df_stop_times['stop_id'].astype(str)).astype(int)

    #Create dataframe of unique dates from the actual data
    df_dates = pd.DataFrame(df['data_frame_ref'].unique(), columns=['data_frame_ref'])

    #Cross join with the stop data, to get the stops for all days
    df_dates['key'] = 1
    df_stop_times['key'] = 1
    df_stop_times = df_dates.merge(df_stop_times, how='outer')
    df_stop_times = df_stop_times.drop('key', axis=1)

    #Merge dataframes together
    df = df_stop_times.merge(df, on=['data_frame_ref', 'trip_id', 'stop_id'], how='left')

    #Create unix time column
    df['stop_time_unix'] = (df['stop_time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    #Interpolate timestamps for missing stop events
    df = df.groupby(['data_frame_ref', 'trip_id']).apply(lambda group: group.interpolate(limit_area='inside'))

    #Convert back to actual timestamps
    df['stop_time'] = pd.to_datetime(df['stop_time_unix'], origin='unix', unit='s')

    #Rename arrival_time and departure_time to prevent future conflicts
    df = df.rename(index=str, columns={"arrival_time": "arrival_time_scheduled", "departure_time": "departure_time_schedule"})

    #Drop uneeeded columns
    df = df[['data_frame_ref', 'trip_id', 'arrival_time_scheduled', 'departure_time_schedule', 'stop_id', 'stop_sequence', 'stop_time', 'stop_time_unix']]

    #Remove NaNs (occurs if we are missing data at the start or end of a journey)
    df = df.dropna(subset=['stop_time'])

    #Reset index
    df = df.reset_index(drop=True)

    return df


def stops_to_durations(df):
    #Get departure and arrival stop info
    df = df[['data_frame_ref', 'trip_id', 'stop_id', 'stop_time', 'stop_time_unix']]
    df_stops_arr = df.copy()
    df = df.rename(index=str, columns={"stop_id": "departure_stop_id", "stop_time": "departure_time", "stop_time_unix": "departure_time_unix"})
    df_stops_arr = df_stops_arr.rename(index=str, columns={"stop_id": "arrival_stop_id", "stop_time": "arrival_time", "stop_time_unix": "arrival_time_unix"})

    #Join the two on trip ID and date
    df = df.merge(df_stops_arr, on=['data_frame_ref', 'trip_id'])

    #Thow out any journeys that do not go forwards in time
    df = df[df['arrival_time_unix'] > df['departure_time_unix']]

    #Add trip duration column
    df['trip_duration'] = df['arrival_time_unix'] - df['departure_time_unix']

    return df

def durations_to_distributions(df):
    #Add hour and minute columns
    df['departure_time_hour'] = df['departure_time'].dt.round('H')
    df['departure_time_minute'] = df['departure_time'].dt.round('min')

    #Get departure and arrival stop info
    df_stops_dep = pd.DataFrame(df['departure_stop_id'].unique(), columns=['departure_stop_id'])
    df_stops_dep['key'] = 1

    df_stops_arr = pd.DataFrame(df['arrival_stop_id'].unique(), columns=['arrival_stop_id'])
    df_stops_arr['key'] = 1

    #Get departure time hours for this dataset
    df_timestamps = pd.DataFrame(df['departure_time_hour'].unique(), columns=['departure_time_hour'])
    df_timestamps['key'] = 1

    #Calculate create minutes array
    df_minutes = pd.DataFrame(np.arange(0,60), columns=['minute'])
    df_minutes['key'] = 1

    #Combine to form base time array
    df_timestamps = df_timestamps.merge(df_minutes).merge(df_stops_dep).merge(df_stops_arr)
    df_timestamps['departure_time_minute'] = df_timestamps['departure_time_hour'] + pd.to_timedelta(df_timestamps.minute, unit='m')
    df_timestamps = df_timestamps[['departure_stop_id', 'arrival_stop_id', 'departure_time_minute']]
    df_timestamps['departure_time_minute_unix'] = (df_timestamps['departure_time_minute'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    #Sort array
    df_timestamps = df_timestamps.sort_values(['departure_stop_id', 'arrival_stop_id', 'departure_time_minute'])
    df_timestamps = df_timestamps.reset_index(drop=True)

    #Join on actual stop data
    df = df_timestamps.merge(df, on=['departure_time_minute', 'departure_stop_id', 'arrival_stop_id'], how='left')

    df.groupby(['departure_stop_id', 'arrival_stop_id']).count()

    #Backfill so each minute has the data for the next departure
    df = df.groupby(['departure_stop_id', 'arrival_stop_id']).apply(lambda group: group.fillna(method='bfill'))

    #Test code
    df[(df['departure_stop_id']==17941) & (df['arrival_stop_id']==16327) & (df['departure_time_minute_unix']>=1541732400)]

    #Add total journey time column
    df['total_journey_time'] = df['arrival_time_unix'] - df['departure_time_minute_unix']

    #Drop NaNs (occurs at the end of the data set when we don't know when the next bus will come.)
    df = df.dropna(subset=['total_journey_time'])

    def calc_distribution(x):
        params = st.gamma.fit(x, floc=True)
        return pd.DataFrame({'shape': [params[0]], 'scale': [params[-1]]})

    df = df.groupby(['departure_time_hour', 'departure_stop_id', 'arrival_stop_id']).agg({'total_journey_time': calc_distribution}).reset_index()
    return df
