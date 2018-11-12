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
        
        #Reset index
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
        df_final = pd.DataFrame(columns=['data_frame_ref', 'trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence', 'stop_time', 'stop_time_unix'])

        #For each day we have data for...
        for data_frame_ref in df_stops['data_frame_ref'].unique():
            print("Processing data_frame_ref {}".format(data_frame_ref))
            df_stops_today = df_stops[df_stops['data_frame_ref']==data_frame_ref]

            #For each trip on that day...
            for trip_id in df_stops_today['journey_ref'].unique():
                print("Processing trip_id {}".format(trip_id))

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
                df_merged = df_merged[['data_frame_ref', 'trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence', 'stop_time', 'stop_time_unix']]

                #Remove NaNs (occurs if we are missing data at the start or end of a journey)
                df_merged = df_merged.dropna(subset=['stop_time'])

                #Add to final data frame
                df_final = pd.concat([df_final, df_merged])

        return df_final
