import json
import numpy as np
import pandas as pd
import pandas_redshift as pr

def raw_to_stops(df):
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

    n_days = len(df_stops['data_frame_ref'].unique())

    #For each day we have data for...
    for i, data_frame_ref in enumerate(df_stops['data_frame_ref'].unique()):
        print("Processing data_frame_ref {} ({} of {})".format(data_frame_ref, (i+1), n_days))
        df_stops_today = df_stops[df_stops['data_frame_ref']==data_frame_ref]

        n_trips = len(df_stops_today['journey_ref'].unique())

        #For each trip on that day...
        for j, trip_id in enumerate(df_stops_today['journey_ref'].unique()):
            print(" Processing trip_id {} ({} of {})".format(trip_id, (j+1), n_trips))

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


def stops_to_durations(df):
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
    n_days = len(df['data_frame_ref'].unique())

    for i, data_frame_ref in enumerate(df['data_frame_ref'].unique()):
        print("Processing data_frame_ref {} ({} of {})".format(data_frame_ref, (i+1), n_days))
        df_today = df[df['data_frame_ref']==data_frame_ref]

        n_dep_stops = len(df_today['stop_id'].unique())

        #For all stops that had departures today...
        for j, departure_stop_id in enumerate(df_today['stop_id'].unique()):
            print(" Processing departure stop_id {} ({} of {})".format(departure_stop_id, (j+1), n_dep_stops))

            #Get trip IDs that stop at this stop, and the departure times from the departure stop of that journey
            df_trip_ids = pd.DataFrame(df_today[df_today['stop_id']==departure_stop_id])
            df_trip_ids = df_trip_ids[['trip_id', 'stop_id', 'stop_time', 'stop_time_unix']]
            df_trip_ids = df_trip_ids.rename(index=str, columns={"stop_id": "departure_stop_id", "stop_time": "departure_time", "stop_time_unix": "departure_time_unix"})

            #Get data for those trip IDs
            df_today_dep = df_trip_ids.merge(df_today, on='trip_id')
            df_today_dep = df_today_dep.rename(index=str, columns={"stop_id": "arrival_stop_id", "stop_time": "arrival_time", "stop_time_unix": "arrival_time_unix"})

            n_arr_stops = len(df_today_dep['arrival_stop_id'].unique())

            #For all stops that had arrivals from this departure stop...
            for k, arrival_stop_id in enumerate(df_today_dep['arrival_stop_id'].unique()):
                print("  Processing arrival stop_id {} ({} of {})".format(arrival_stop_id, (k+1), n_arr_stops))

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

def connect_to_redshift():
    with open('credentials.json') as json_data:
        credentials = json.load(json_data)

    #Connect to Redshift
    pr.connect_to_redshift(dbname = 'muni',
                        host = 'jonobate.c9xvjgh0xspr.us-east-1.redshift.amazonaws.com',
                        port = '5439',
                        user = credentials['user'],
                        password = credentials['password'])

def connect_to_s3():
    with open('credentials.json') as json_data:
        credentials = json.load(json_data)

    pr.connect_to_s3(aws_access_key_id = credentials['aws_access_key_id'],
                aws_secret_access_key = credentials['aws_secret_access_key'],
                bucket = 'jonobate-bucket')

if __name__ == '__main__':
    #Get raw data from processing
    connect_to_redshift()
    print('Getting vehicle_monitoring data from Redshift...')
    df = pr.redshift_to_pandas("""select * from vehicle_monitoring
                                where data_frame_ref not in (select distinct data_frame_ref from stop_events)
                                and data_frame_ref < trunc(convert_timezone('US/Pacific', GETDATE()));""")
    pr.close_up_shop()

    #Parse into stop events
    df = raw_to_stops(df)

    #Write results to stop_events
    connect_to_s3()
    connect_to_redshift()
    print('Writing stop_events data to Redshift...')
    pr.pandas_to_redshift(data_frame = df,
                        redshift_table_name = 'stop_events',
                        append = True)

    #Get stop events for processing
    print('Getting stop_events data from Redshift...')
    df = pr.redshift_to_pandas("""select * from stop_events
                                where data_frame_ref not in (select distinct data_frame_ref from trip_durations)
                                and data_frame_ref < trunc(convert_timezone('US/Pacific', GETDATE()));""")

    pr.close_up_shop()

    #Parse into journey durations
    df = stops_to_durations(df)

    #Write results to stop_events
    connect_to_s3()
    connect_to_redshift()
    print('Writing trip_durations data to Redshift...')
    pr.pandas_to_redshift(data_frame = df,
                        redshift_table_name = 'trip_durations',
                        append = True)

    pr.close_up_shop()
