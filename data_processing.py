import json
import numpy as np
import pandas as pd
import pandas_redshift as pr
import pickle
import time
from os import listdir
from os.path import isfile, join
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import interp1d
import scipy.stats as st

def get_raw(sample_flag):
    with open('credentials.json') as json_data:
        credentials = json.load(json_data)

    pr.connect_to_redshift(dbname = 'muni',
                        host = 'jonobate.c9xvjgh0xspr.us-east-1.redshift.amazonaws.com',
                        port = '5439',
                        user = credentials['user'],
                        password = credentials['password'])

    if sample_flag:
        df = pr.redshift_to_pandas("""select * from vehicle_monitoring limit 1000""")
        df.to_csv('data/vehicle_monitoring_sample.csv', index=False)
    else:
        df = pr.redshift_to_pandas("""select * from vehicle_monitoring""")
        df.to_csv('data/vehicle_monitoring.csv', index=False)
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


def create_features(df, df_gtfs):
    #Convert timestamps
    df['departure_time_hour'] = pd.to_datetime(df['departure_time_hour'])

    #Re-localize time
    df['departure_time_hour'] = df['departure_time_hour'].dt.tz_localize('utc').dt.tz_convert('US/Pacific')

    #Make sure stop IDs are minutes
    df['departure_stop_id'] = df['departure_stop_id'].astype(int)
    df['arrival_stop_id'] = df['arrival_stop_id'].astype(int)

    #Generate local day of week and hour features
    df['dow'] = df['departure_time_hour'].dt.dayofweek
    df['hour'] = df['departure_time_hour'].dt.hour

    #Append stop metadata
    df_dep = df_gtfs['stops'].copy()
    df_dep = df_dep.add_suffix('_dep')

    df_arr = df_gtfs['stops'].copy()
    df_arr = df_arr.add_suffix('_arr')

    df = df.merge(df_dep, left_on='departure_stop_id', right_on='stop_id_dep')
    df = df.merge(df_arr, left_on='arrival_stop_id', right_on='stop_id_arr')

    #Add route information. Outer join to help with get_dummies.
    df_routes = df_gtfs['routes'].copy()
    df_routes['route_short_name'] = 'Muni-' + df_routes['route_short_name'].astype(str)
    df = df.merge(df_routes, on='route_short_name', how='outer')

    #Calculate stop distances
    df['stop_lat_dist'] = (df['stop_lat_dep'] - df['stop_lat_arr'])
    df['stop_lon_dist'] = (df['stop_lon_dep'] - df['stop_lon_arr'])
    df['stop_dist'] = np.sqrt((df['stop_lat_dist']**2) + (df['stop_lon_dist']**2))

    #Create dummies from stop names
    df = pd.get_dummies(df, columns=['route_short_name'])

    #Drop null columns, drop string/datetime columns
    df = df.dropna(axis='columns', how='all')
    df = df.drop(df.select_dtypes(['object', 'datetime64[ns]', 'datetime64[ns, US/Pacific]']), axis=1)

    #Drop rows with nulls (gets rid of the rows added for dummying purposes)
    df = df.dropna(axis='rows', how='any')

    #Reset index
    df = df.reset_index(drop=True)
    return df

def fit_default(X, y, name, sample_flag):

    clf = RandomForestRegressor(oob_score=True)

    clf.fit(X, y)

    # Save the  model to a pickle file
    if sample_flag:
        pickle.dump(clf, open('{}_sample.pickle'.format(name), 'wb'))
    else:
        pickle.dump(clf, open('{}.pickle'.format(name), 'wb'))

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
        pickle.dump(clf.best_estimator_, open('{}_sample.pickle'.format(name), 'wb'))
    else:
        pickle.dump(clf.best_estimator_, open('{}.pickle'.format(name), 'wb'))

    return clf.best_estimator_


def raw_to_stops(df, df_gtfs):
    #Give the GTFS dictionary dataframes aliases to avoid confusion later
    df_stop_times = df_gtfs['stop_times']
    df_trips = df_gtfs['trips']
    df_routes = df_gtfs['routes']

    df_offset = df_stop_times.copy()

    #Calculate offset for calculating reference day
    df_offset[['hours', 'mins', 'secs']] = df_offset['departure_time'].str.split(':', n=2, expand=True).astype(int)
    df_offset['time'] = df_offset['hours'] + (df_offset['mins']/60) + (df_offset['secs']/(60*60))
    df_offset['offset'] = 12 - df_offset['time']
    df_offset = df_offset.groupby('trip_id').mean().reset_index()[['trip_id', 'offset']]
    df_offset['offset'] = pd.to_timedelta(df_offset['offset'], unit='h')

    #Drop uneeded columns
    df = df[['line_ref', 'journey_ref', 'recorded_time', 'stop_point_ref']]

    #Convert timestamps
    df['recorded_time'] = pd.to_datetime(df['recorded_time'])

    #Rename columns to match GTFS data
    df = df.rename(index=str, columns={"line_ref": "route_short_name", "journey_ref": "trip_id", "stop_point_ref": "stop_id"})

    #Fix to deal with the fact that that stop_ids and trip_ids are in a slightly different format in the raw data
    df['stop_id'] = df['stop_id'].astype(str).str[1:].astype(int)

    #Merge dataframes
    df = df.merge(df_offset, on='trip_id', how='left')

    #Fill NAs (would happen if there was no trip ID in the GTFS data... assume zero offset)
    df['offset'] = df['offset'].fillna(0)

    #Calculate schedule date
    df['schedule_date'] = (df['recorded_time'] + df['offset']).dt.round('D')

    #Drop uneeded columns
    df = df[['schedule_date', 'route_short_name', 'trip_id', 'stop_id', 'recorded_time']]

    #Sort values, reset index
    df = df.sort_values(['schedule_date', 'trip_id', 'recorded_time'])
    df = df.reset_index(drop=True)

    #Create offset dataframe with next poll data
    df_next = df.copy()
    df_next = df_next.add_suffix('_next')

    #Add indexes to original and offset data
    df['join_index'] = df.index.astype(int)
    df_next['join_index'] = df_next.index
    df_next['join_index'] = df_next['join_index'].astype(int) - 1

    #Join data to offset data
    df = df.merge(df_next, on='join_index')

    #Filter to stop events
    df = df[(df['schedule_date']==df['schedule_date_next'])
          & (df['trip_id']==df['trip_id_next'])
          & (df['stop_id']!=df['stop_id_next'])]

    #Add in stop time column
    df['stop_time'] = df['recorded_time'] + (df['recorded_time_next'] - df['recorded_time']) / 2

    #Create dataframe of unique dates from the actual data
    df_dates = pd.DataFrame(df['schedule_date'].unique(), columns=['schedule_date'])

    #Compile dataset
    df_stop_data = df_stop_times.merge(df_trips, on='trip_id').merge(df_routes, on='route_id')
    df_stop_data = df_stop_data[['route_short_name', 'trip_id', 'stop_id', 'arrival_time', 'departure_time', 'stop_sequence']]

    #Cross join with the stop data, to get the stops for all days
    df_dates['key'] = 1
    df_stop_data['key'] = 1
    df_stop_data = df_dates.merge(df_stop_data, how='outer')
    df_stop_data = df_stop_data.drop('key', axis=1)

    #Fix to deal with bug where pandas treats line names as ints - add some text to the route_short_name
    df['route_short_name'] = ('Muni-'+df['route_short_name'].astype(str))
    df_stop_data['route_short_name'] = ('Muni-'+df_stop_data['route_short_name'].astype(str))

    #Merge dataframes together
    df = df_stop_data.merge(df, on=['schedule_date', 'route_short_name', 'trip_id', 'stop_id'], how='outer')

    #Create unix time column
    df['stop_time_unix'] = (df['stop_time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    #Interpolate timestamps for missing stop events
    df['stop_time_unix'] = df.groupby(['schedule_date', 'trip_id'])['stop_time_unix'].apply(lambda group: group.interpolate(limit_area='inside'))

    #Convert back to actual timestamps
    df['stop_time'] = pd.to_datetime(df['stop_time_unix'], origin='unix', unit='s')

    #Drop uneeeded columns
    df = df[['schedule_date', 'route_short_name', 'trip_id', 'stop_id', 'stop_time', 'stop_time_unix']]

    #Remove NaNs (occurs if we are missing data at the start or end of a journey)
    df = df.dropna(subset=['stop_time_unix'])

    #Reset index
    df = df.reset_index(drop=True)

    return df


def stops_to_durations(df):
    #Get departure and arrival stop info
    df_stops_arr = df.copy()
    df = df.rename(index=str, columns={"stop_id": "departure_stop_id", "stop_time": "departure_time", "stop_time_unix": "departure_time_unix"})
    df_stops_arr = df_stops_arr.rename(index=str, columns={"stop_id": "arrival_stop_id", "stop_time": "arrival_time", "stop_time_unix": "arrival_time_unix"})

    #Join the two on trip ID and date
    df = df.merge(df_stops_arr, on=['schedule_date', 'route_short_name', 'trip_id'])

    #Thow out any journeys that do not go forwards in time
    df = df[df['arrival_time_unix'] > df['departure_time_unix']]

    #Add trip duration column
    df['trip_duration'] = df['arrival_time_unix'] - df['departure_time_unix']

    #Add hour and minute columns
    df['departure_time_hour'] = df['departure_time'].dt.floor('H')
    df['departure_time_minute'] = df['departure_time'].dt.floor('min')

    return df
'''
def durations_to_distributions(df, start_time):
    #Create minutes array
    df_minutes = pd.DataFrame(np.arange(0,60), columns=['minute'])
    df_minutes['key'] = 1

    df_final = pd.DataFrame(columns=['schedule_date', 'route_short_name', 'departure_time_hour','departure_stop_id','arrival_stop_id','shape','scale','mean'])

    #Looping to avoid memory errors. Get unique combos to loop through.
    iter_list = df.groupby(['schedule_date', 'route_short_name', 'departure_time_hour']).count().reset_index()[['schedule_date', 'route_short_name', 'departure_time_hour']]
    iter_len = iter_list.shape[0]

    #Loop through rows
    for i, row in iter_list.iterrows():
        #i = 1
        #row = iter_list.iloc[i]

        date = row['schedule_date']
        route = row['route_short_name']
        hour = row['departure_time_hour']
        next_hour = hour + pd.to_timedelta(1, unit='h')

        print('Processing row {} of {}, {} secs elapse'.format(i+1, iter_len, time.time() - start_time))

        #Create subset of data for hour and route in question. Also get data for next hour.
        df_temp = df[(df['schedule_date']==date) & (df['route_short_name']==route) & ((df['departure_time_hour']==hour) |  (df['departure_time_hour']==next_hour))]

        #Get stop pairs from data
        df_timestamps = df_temp.groupby(['departure_time_hour', 'departure_stop_id', 'arrival_stop_id']).count().reset_index()[['departure_time_hour', 'departure_stop_id', 'arrival_stop_id']]
        df_timestamps['key'] = 1

        #Merge with minutes
        df_timestamps = df_timestamps.merge(df_minutes, on='key')

        #Create minutes timestamps and unix time column
        df_timestamps['departure_time_minute'] = hour + pd.to_timedelta(df_timestamps.minute, unit='m')
        df_timestamps['departure_time_minute_unix'] = (df_timestamps['departure_time_minute'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        #Drop uneeded columns
        df_timestamps = df_timestamps[['departure_time_hour', 'departure_stop_id', 'arrival_stop_id', 'departure_time_minute', 'departure_time_minute_unix']]

        #Sort array
        df_timestamps = df_timestamps.sort_values(['departure_stop_id', 'arrival_stop_id', 'departure_time_minute'])

        #Join on actual stop data
        df_temp = df_timestamps.merge(df_temp, on=['departure_time_hour', 'departure_time_minute', 'departure_stop_id', 'arrival_stop_id'], how='left')

        del(df_timestamps)

        #Backfill so each minute has the data for the next departure
        df_temp = df_temp.groupby(['departure_stop_id', 'arrival_stop_id']).apply(lambda group: group.fillna(method='bfill'))

        #Drop second hour - no longer needed now that we have done the Backfill
        df_temp = df_temp[df_temp['departure_time_hour']==hour]

        #Add total journey time column
        df_temp['total_journey_time'] = df_temp['arrival_time_unix'] - df_temp['departure_time_minute_unix']

        #Drop NaNs (occurs at the end of the data set when we don't know when the next bus will come.)
        df_temp = df_temp.dropna(subset=['total_journey_time'])

        def calc_distribution(x):
            try:
                params = st.gamma.fit(x[x > 0], floc=0)
                shape = params[0]
                scale = params[2]
            except Exception as e:
                print(e)
                print(x)
                shape = np.NaN
                scale = np.NaN
            return shape, scale

        #Calculate shape and scale parameters
        df_temp = df_temp.groupby(['departure_time_hour', 'departure_stop_id', 'arrival_stop_id'])['total_journey_time'].agg(calc_distribution).reset_index()

        #Split into columns
        df_temp[['shape', 'scale']] = df_temp['total_journey_time'].apply(pd.Series)

        #Generate Target
        df_temp['mean'] = df_temp['shape'] * df_temp['scale']

        df_temp['schedule_date'] = date
        df_temp['route_short_name'] = route

        #Drop uneeded columns
        df_temp = df_temp[['schedule_date', 'route_short_name', 'departure_time_hour','departure_stop_id','arrival_stop_id','shape','scale','mean']]

        #Drop NAs
        df_temp = df_temp.dropna()

        df_final = df_final.append(df_temp)

        del(df_temp)

    return df_final
'''

def durations_to_distributions(df):
    #Add hour and minute columns
    df['departure_time_hour'] = df['departure_time'].dt.floor('H')
    df['departure_time_minute'] = df['departure_time'].dt.floor('min')

    #Get departure and arrival stop info
        #Get stop pairs from data
    df_timestamps = df.groupby(['schedule_date', 'route_short_name', 'departure_time_hour', 'departure_stop_id', 'arrival_stop_id']).count().reset_index()[['schedule_date', 'route_short_name', 'departure_time_hour', 'departure_stop_id', 'arrival_stop_id']]
    df_timestamps['key'] = 1

    #Create minutes array
    df_minutes = pd.DataFrame(np.arange(0,60), columns=['minute'])
    df_minutes['key'] = 1

    #Combine to form base time array
    df_timestamps = df_timestamps.merge(df_minutes)
    del(df_minutes)

    df_timestamps['departure_time_minute'] = df_timestamps['departure_time_hour'] + pd.to_timedelta(df_timestamps.minute, unit='m')
    df_timestamps['departure_time_minute_unix'] = (df_timestamps['departure_time_minute'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    df_timestamps = df_timestamps[['schedule_date', 'route_short_name', 'departure_time_hour', 'departure_stop_id', 'arrival_stop_id', 'departure_time_minute', 'departure_time_minute_unix']]

    #Sort array
    df_timestamps = df_timestamps.sort_values(['departure_stop_id', 'arrival_stop_id', 'departure_time_minute'])
    df_timestamps = df_timestamps.reset_index(drop=True)

    #Join on actual stop data
    df = df_timestamps.merge(df, on=['schedule_date', 'route_short_name', 'departure_time_hour', 'departure_time_minute', 'departure_stop_id', 'arrival_stop_id'], how='left')
    del(df_timestamps)

    #Backfill so each minute has the data for the next departure
    df = df.groupby(['route_short_name','departure_stop_id', 'arrival_stop_id']).apply(lambda group: group.fillna(method='bfill'))

    #Add total journey time column
    df['total_journey_time'] = df['arrival_time_unix'] - df['departure_time_minute_unix']

    #Drop NaNs (occurs at the end of the data set when we don't know when the next bus will come.)
    df = df.dropna(subset=['total_journey_time'])

    def calc_distribution(x):
        try:
            params = st.gamma.fit(x[x > 0], floc=0)
            shape = params[0]
            scale = params[2]
        except Exception as e:
            print(e)
            print(x)
            shape = np.NaN
            scale = np.NaN
        return shape, scale

    #Calculate shape and scale parameters
    df = df.groupby(['schedule_date', 'route_short_name', 'departure_time_hour', 'departure_stop_id', 'arrival_stop_id'])['total_journey_time'].agg(calc_distribution).reset_index()

    #Split into columns
    df[['shape', 'scale']] = df['total_journey_time'].apply(pd.Series)

    #Generate Target
    df['mean'] = df['shape'] * df['scale']

    #Drop uneeded columns
    df = df[['schedule_date', 'route_short_name', 'departure_time_hour', 'departure_stop_id','arrival_stop_id','shape','scale','mean']]

    #Drop NAs
    df = df.dropna()

    return df
