import json
import numpy as np
import pandas as pd
import pandas_redshift as pr
import pickle
from os import listdir
from os.path import isfile, join
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def get_distributions():
    with open('credentials.json') as json_data:
        credentials = json.load(json_data)

    pr.connect_to_redshift(dbname = 'muni',
                        host = 'jonobate.c9xvjgh0xspr.us-east-1.redshift.amazonaws.com',
                        port = '5439',
                        user = credentials['user'],
                        password = credentials['password'])

    df_dists = pr.redshift_to_pandas("""select *,
                                    convert_timezone('US/Pacific', departure_time_hour) as local_departure_time_hour
                                     from distributions_gamma""")
    pr.close_up_shop()


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

    # Params to pass to the GridSearchCV
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['mse', 'mae']
    }

    model = RandomForestRegressor()

    # Create the GridSearch model
    clf = GridSearchCV(model, param_grid, cv=5, verbose=10, n_jobs=-1)

    # Fit the GridSearch model
    clf.fit(X, y)

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


class RawToStops(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def raw_to_stops(self,X):
        df = X

        #Load stop data
        df_stop_times = pd.read_csv('google_transit/stop_times.txt')

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

        #Create offset dataframe
        df_next = df[['data_frame_ref', 'journey_ref', 'recorded_time', 'stop_point_ref', 'stop_point_name']]
        df_next = df_next.add_suffix('_next')
        df_next['join_index'] = df_next.index
        df_next['join_index'] = df_next['join_index'].astype(int) - 1

        #Join data to offset data
        df_stops = df.merge(df_next, on='join_index')

        #Filter to stop events
        df_stops = df_stops[(df_stops['data_frame_ref']==df_stops['data_frame_ref_next'])
              & (df_stops['journey_ref']==df_stops['journey_ref_next'])
              & (df_stops['stop_point_ref']!=df_stops['stop_point_ref_next'])]

        #Add in stop time column
        df_stops['stop_time'] = df_stops['recorded_time'] + (df_stops['recorded_time_next'] - df_stops['recorded_time']) / 2

        #Drop uneeded columns
        df_stops = df_stops[['data_frame_ref', 'journey_ref', 'stop_point_ref', 'stop_time']]

        #Create output dataframe
        df_final = pd.DataFrame(columns=['data_frame_ref', 'trip_id', 'stop_id', 'stop_time', 'stop_time_unix'])

        #For each day we have data for...
        for data_frame_ref in df_stops['data_frame_ref'].unique():
            print("Processing data_frame_ref {}".format(data_frame_ref))
            df_stops_today = df_stops[df_stops['data_frame_ref']==data_frame_ref]

            #For each trip on that day...
            for trip_id in df_stops_today['journey_ref'].unique():
                print(" Processing trip_id {}".format(trip_id))

                #Get actual data for this trip. Rename columns to match stop data.
                df_stops_actual = df_stops_today[df_stops_today['journey_ref']==trip_id].rename(index=str, columns={"journey_ref": "trip_id", "stop_point_ref": "stop_id"})

                #Get stop data for this trip
                df_stops_all = df_stop_times[df_stop_times['trip_id'] == trip_id]

                #Fix to deal with the fact that that stop_ids are in a slightly different format
                df_stops_all['stop_id'] = ('1' + df_stops_all['stop_id'].astype(str)).astype(int)

                #Merge dataframes todether
                df_merged = df_stops_all.merge(df_stops_actual, on=['trip_id', 'stop_id'], how='left')

                #Create unix time column
                df_merged['stop_time_unix'] = (df_merged['stop_time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

                #Interpolate timestamps for missing stop events
                df_merged['stop_time_unix'] = df_merged['stop_time_unix'].interpolate(limit_area='inside')

                #Convert back to actual timestamps
                df_merged['stop_time'] = pd.to_datetime(df_merged['stop_time_unix'], origin='unix', unit='s')

                #Fill missing data_frame_refs
                df_merged['data_frame_ref'] = df_merged['data_frame_ref'].fillna(data_frame_ref)

                #Drop uneeeded columns
                df_merged = df_merged[['data_frame_ref', 'trip_id', 'stop_id','stop_time', 'stop_time_unix']]

                #Remove NaNs (occurs if we are missing data at the start or end of a journey)
                df_merged = df_merged.dropna(subset=['stop_time'])

                #Add to final data frame
                df_final = pd.concat([df_final, df_merged])

        return df_final


class StopsToDurations(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self,X):
        df = X

        #Create output dataframe
        df_final = pd.DataFrame(columns=['data_frame_ref',
                                         'trip_id',
                                         'departure_stop_id',
                                         'departure_time',
                                         'departure_time_unix',
                                         'arrival_stop_id',
                                         'arrival_time',
                                         'arrival_time_unix',
                                         'trip_duration'])

        #For each day we have data for...
        for data_frame_ref in df['data_frame_ref'].unique():
            print("Processing data_frame_ref {}".format(data_frame_ref))
            df_today = df[df['data_frame_ref']==data_frame_ref]

            #For all stops that had departures today...
            for departure_stop_id in df_today['stop_id'].unique():
                print(" Processing departure stop_id {}".format(departure_stop_id))

                #Get trip IDs that stop at this stop, and the departure times from the departure stop of that journey
                df_trip_ids = pd.DataFrame(df_today[df_today['stop_id']==departure_stop_id])
                df_trip_ids = df_trip_ids[['trip_id', 'stop_id', 'stop_time', 'stop_time_unix']]
                df_trip_ids = df_trip_ids.rename(index=str, columns={"stop_id": "departure_stop_id", "stop_time": "departure_time", "stop_time_unix": "departure_time_unix"})

                #Get data for those trip IDs
                df_today_dep = df_trip_ids.merge(df_today, on='trip_id')
                df_today_dep = df_today_dep.rename(index=str, columns={"stop_id": "arrival_stop_id", "stop_time": "arrival_time", "stop_time_unix": "arrival_time_unix"})

                #For all stops that had arrivals from this departure stop...
                for arrival_stop_id in df_today_dep['arrival_stop_id'].unique():
                    print("  Processing arrival stop_id {}".format(arrival_stop_id))

                    #Filter to that stop, but only if arrival time is later than departure time
                    df_today_dep_arr = df_today_dep[(df_today_dep['arrival_stop_id']==arrival_stop_id) & (df_today_dep['departure_time_unix'] < df_today_dep['arrival_time_unix'])]
                    df_today_dep_arr['trip_duration'] = df_today_dep_arr['arrival_time_unix'] - df_today_dep_arr['departure_time_unix']

                    #Reorder because it makes me happy...
                    df_today_dep_arr = df_today_dep_arr[['data_frame_ref',
                                                         'trip_id',
                                                         'departure_stop_id',
                                                         'departure_time',
                                                         'departure_time_unix',
                                                         'arrival_stop_id',
                                                         'arrival_time',
                                                         'arrival_time_unix',
                                                         'trip_duration']]

                    #Add to final data frame
                    df_final = pd.concat([df_final, df_today_dep_arr])

        return df_final
