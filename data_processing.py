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
from datetime import timedelta

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


def grid_search(X, y, name, sample_flag):
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
    if sample_flag:
        pickle.dump(clf.best_estimator_, open('{}_sample.pickle'.format(name), 'wb'))
    else:
        pickle.dump(clf.best_estimator_, open('{}.pickle'.format(name), 'wb'))

    return clf


def raw_to_stops(df, gtfs_fn, timezone="America/Los_Angeles"):
    """
    Convert Muni API raw responses ("GPS fixes") into stop events. This is a 
    tricky process, because successive GPS fixes may span a period of time 
    during which the vehicle passed more than one stop.

                fix1                    fix2       
    +------------+-----------------------+---------->
    +----|-----------|--------|------|--------|----->
        stop1      stop2    stop3   stop4    stop5
                      --> time

    Consequently, fancy interpolation is necessary.
    
    Args:
        df (DataFrame): Each row is a response from the Muni vehicle 
            information API; (ie, a "GPS fix"). Each GPS fix has columns 
            describing the current date and time, some information about 
            vehicle and the route it's on, its location, and its predicted 
            arrival at the next stop.
        gtfs_fn (string): Relative name of directory containing unzipped GTFS 
            feed. 
        timezone (string): standard time zone in which data was generated.

    Returns:
        (DataFrame): Each row is a "stop passby" event, detailing an event in 
            which a vehicle running a particular trip passed by a particular 
            stop. Each row contains information about the service day, trip, 
            stop, and the time of the event. 
    """

    df = df.copy()

    # Convert time columns into tz-aware datetimes
    for colname in ["recorded_time","valid_until_time",
                    "expected_arrival_time","expected_departure_time"]:
        df[colname] = pd.to_datetime( df[colname], utc=True ).dt.tz_convert(timezone)
    
    # no need to convert data_frame_ref's time zone; it's just an unqualified date
    df["data_frame_ref"] = pd.to_datetime( df["data_frame_ref"] )


    df_stop_times = pd.read_csv( join( gtfs_fn, 'stop_times.txt') )

    # The column "date_time_ref" corresponds to the "service day" of the GPS fix. For times
    # before 4 am, the service day is actually the day before. For example, a vehicle
    # operating at 3am on November 11 is running accordingto the November 10 schedule.
    df.loc[ df.expected_departure_time.dt.hour<4 , "data_frame_ref"] -= timedelta(days=1)

    #Drop uneeded columns
    df = df[['data_frame_ref', 'line_ref', 'journey_ref', 'recorded_time', 'stop_point_ref']]

    #Rename columns to match GTFS data
    df = df.rename(index=str, columns={"line_ref": "route_short_name", 
                                "journey_ref": "trip_id", 
                                "stop_point_ref": "stop_id",
                                "data_frame_ref": "schedule_date"})

    #Fix to deal with the fact that that stop_ids and trip_ids are in a slightly different format in the raw data
    df['stop_id'] = df['stop_id'].astype(str).str[1:].astype(int)

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
    df_stop_data = df_stop_times[['trip_id', 'stop_id', 'arrival_time', 'departure_time', 'stop_sequence']]

    #Cross join with the stop data, to get the stops for all days
    df_dates['key'] = 1
    df_stop_data['key'] = 1
    df_stop_data = df_dates.merge(df_stop_data, how='outer')
    df_stop_data = df_stop_data.drop('key', axis=1)

    #Merge dataframes together
    df = df_stop_data.merge(df, on=['schedule_date', 'trip_id', 'stop_id'], how='outer')

    #Create unix time column
    df['stop_time_unix'] = epoch_seconds( df['stop_time'], preserve_null=True )

    #Interpolate timestamps for missing stop events
    df['stop_time_unix'] = df.groupby(['schedule_date', 'trip_id'])['stop_time_unix'].apply(lambda group: group.interpolate(limit_area='inside'))

    #Convert back to actual timestamps
    df['stop_time'] = pd.to_datetime(df['stop_time_unix'], unit='s', utc=True).dt.tz_convert(timezone)

    #Drop uneeeded columns
    df = df[['schedule_date', 'trip_id', 'stop_id', 'stop_time']]

    #Remove NaNs (occurs if we are missing data at the start or end of a journey)
    df = df.dropna(subset=['stop_time'])

    #Reset index
    df = df.reset_index(drop=True)

    return df


def stops_to_durations(df):
    """
    Finds all pairs of bus pass-by events where the second event is after
    the first and both stops are on the same trip instance.

    Args:
        df (DataFrame): Contains bus pass-bys; each row contains the 
        schedule_date, trip_id, stop_id, and time of the pass-by event.

    Returns:
        (DataFrame): Each row contains information on the journey time
        between a pair of stops on a trip instance. The total nummber of rows 
        returned will be k*p^2, where k is the number of trip instances and p 
        is the average number of stops per trip instance.
    """

    #Get departure and arrival stop info
    df_stops_arr = df.copy()
    df = df.rename(index=str, columns={"stop_id": "departure_stop_id", 
                                    "stop_time": "departure_time"})
    df_stops_arr = df_stops_arr.rename(index=str, columns={"stop_id": "arrival_stop_id", 
                    "stop_time": "arrival_time"})

    #Join the two on trip ID and date
    df = df.merge(df_stops_arr, on=['schedule_date', 'trip_id'])

    #Thow out any journeys that do not go forwards in time
    df = df[df['arrival_time'] > df['departure_time']]

    #Add trip duration column
    df['trip_duration'] = (df['arrival_time'] - df['departure_time']).dt.total_seconds()

    return df

def cartesian_product( lsts ):
    """
    Returns Pandas DataFrame containing cartesian product of lists. This is the 
    same as itertools.product, but faster.
    """

    ret = None

    for lst in lsts:
        subtable = pd.DataFrame(lst)
        subtable["key"] = 1

        if ret is None:
            ret = subtable
        else:
            ret = ret.merge(subtable, on="key")

    # they 'key' column was just a trick to get a set product; it's no longer needed
    ret = ret.drop("key", axis=1)

    return ret

def epoch_seconds( timestamp_series, preserve_null=True ):
    if preserve_null:
        return (timestamp_series.dt.tz_convert("utc") - pd.Timestamp("1970-01-01", tz="utc")) // pd.Timedelta('1s')
    else:
        return timestamp_series.astype(np.int64) // 1e9

def durations_to_distributions(df):
    """
    Finds parameter estimates for the distribution of travel times for all 
    sets of (start_time, route_name, origin_stop, destination_stop) present in
    the input dataframe.

    Args:
        df (DataFrame): DataFrame in format returned by `stops_to_durations`.

    Returns:
        (DataFrame): Contains distribution parameters of fit beta 
        distribution for all (start_time, route_name, origin_stop, 
        destination_stop) present in `df`.
    """

    #Add hour and minute columns
    df['departure_time_hour'] = df['departure_time'].dt.round('H')
    df['departure_time_minute'] = df['departure_time'].dt.round('min')

    # we'll construct a dataframe with those unique combinations of origin
    # stop, destination stop, and departure time
    df_timestamps = cartesian_product( [df['departure_stop_id'].unique(),
                              df['arrival_stop_id'].unique(),
                              df['departure_time_hour'].unique(),
                              np.arange(0,60)] )
    df_timestamps.columns = ["departure_stop_id", "arrival_stop_id", "departure_time_hour", "minute"]
    # the `departure_time_hour` and `minute` columns were just a means towards
    # a `departure_time_minute` column
    df_timestamps['departure_time_minute'] = df_timestamps['departure_time_hour'] + pd.to_timedelta(df_timestamps.minute, unit='m')
    df_timestamps.drop( ["departure_time_hour", "minute"], axis=1, inplace=True )
    df_timestamps['departure_time_minute_unix'] = epoch_seconds( df_timestamps['departure_time_minute'] )

    #Sort array
    df_timestamps = df_timestamps.sort_values(['departure_stop_id', 'arrival_stop_id', 'departure_time_minute'])
    df_timestamps = df_timestamps.reset_index(drop=True)

    #Join on actual stop data
    df = df_timestamps.merge(df, on=['departure_time_minute', 'departure_stop_id', 'arrival_stop_id'], how='left')



    #Backfill so each minute has the data for the next departure
    df = df.groupby(['departure_stop_id', 'arrival_stop_id']).apply(lambda group: group.fillna(method='bfill'))

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
    df = df.groupby(['departure_time_hour', 'route_short_name', 'departure_stop_id', 'arrival_stop_id'])['total_journey_time'].agg(calc_distribution).reset_index()

    #Split into columns
    df[['shape', 'scale']] = df['total_journey_time'].apply(pd.Series)
    df = df.drop('total_journey_time', axis=1)

    #Drop NAs
    df = df.dropna()

    #Generate Target
    df['mean'] = df['shape'] * df['scale']

    return df
