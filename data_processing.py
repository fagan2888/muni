import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RawToStops(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self,X):
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