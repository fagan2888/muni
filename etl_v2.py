import json
import numpy as np
import pandas as pd
import pandas_redshift as pr


def raw_to_stops():
    connect_to_redshift()
    connect_to_s3()

    #Load stop data
    df_stop_times = pd.read_csv('google_transit/stop_times.txt')

    print('Getting vehicle_monitoring data from Redshift...')
    df = pr.redshift_to_pandas("""select data_frame_ref
                                from vehicle_monitoring
                                where data_frame_ref not in (select distinct data_frame_ref from stop_events)
                                and data_frame_ref < trunc(convert_timezone('US/Pacific', GETDATE()))
                                group by data_frame_ref""")

    n_days = df.shape[0]

    for i, row in df.iterrows():
        data_frame_ref = row['data_frame_ref']
        print("Processing data_frame_ref {} ({} of {})".format(data_frame_ref, (i+1), n_days))

        df_cur = pr.redshift_to_pandas("""select * from vehicle_monitoring
                                where data_frame_ref = '{}';""".format(data_frame_ref))

        #Only bother with this if we actually have data...
        if df_cur.shape[0] == 0:
            print("No data for {}, skipping...".format(data_frame_ref))
        else:
            #Convert datetimes
            df_cur['recorded_time'] = pd.to_datetime(df_cur['recorded_time'])
            df_cur['valid_until_time'] = pd.to_datetime(df_cur['valid_until_time'])
            df_cur['data_frame_ref'] = pd.to_datetime(df_cur['data_frame_ref'])
            df_cur['expected_arrival_time'] = pd.to_datetime(df_cur['expected_arrival_time'])
            df_cur['expected_departure_time'] = pd.to_datetime(df_cur['expected_departure_time'])

            #Sort values, reset index
            df_cur = df_cur.sort_values(['data_frame_ref', 'journey_ref', 'recorded_time'])
            df_cur = df_cur.reset_index(drop=True)
            df_cur['join_index'] = df_cur.index.astype(int)

            #Create offset dataframe
            df_next = df_cur[['data_frame_ref', 'journey_ref', 'recorded_time', 'stop_point_ref', 'stop_point_name']]
            df_next = df_next.add_suffix('_next')
            df_next['join_index'] = df_next.index
            df_next['join_index'] = df_next['join_index'].astype(int) - 1

            #Join data to offset data
            df_stops = df_cur.merge(df_next, on='join_index')

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

            n_trips = len(df_stops['journey_ref'].unique())

            #For each trip on that day...
            for j, trip_id in enumerate(df_stops['journey_ref'].unique()):
                print(" Processing trip_id {} ({} of {})".format(trip_id, (j+1), n_trips))

                #Get actual data for this trip. Rename columns to match stop data.
                df_stops_actual = df_stops[df_stops['journey_ref']==trip_id].rename(index=str, columns={"journey_ref": "trip_id", "stop_point_ref": "stop_id"})

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

            #Only bother with this if we actually have stop events...
            if df_final.shape[0] == 0:
                print("No stop events for {}, skipping...".format(data_frame_ref))
            else:
                pr.pandas_to_redshift(data_frame = df_final,
                                    redshift_table_name = 'stop_events',
                                    append = True)

    pr.close_up_shop()


def stops_to_durations():

    connect_to_redshift()

    df = pr.redshift_to_pandas("""select a.* from
        (select data_frame_ref, stop_id from stop_events group by data_frame_ref, stop_id) a
        left join
        (select data_frame_ref, departure_stop_id from trip_durations group by data_frame_ref, departure_stop_id) b
        on a.data_frame_ref = b.data_frame_ref
        	and a.stop_id = b.departure_stop_id
        where b.data_frame_ref is null
        	and b.departure_stop_id is null
            and a.data_frame_ref < trunc(convert_timezone('US/Pacific', GETDATE()))
            order by a.data_frame_ref, a.stop_id;""")

    n_days_dep_stops = df.shape[0]

    for i, row in df.iterrows():
        data_frame_ref = row['data_frame_ref']
        dep_stop_id = row['stop_id']
        print("Processing data_frame_ref {}, departure_stop_id {} ({} of {})".format(data_frame_ref, dep_stop_id, (i+1), n_days_dep_stops))

        pr.exec_commit("""insert into trip_durations
            select a.data_frame_ref,
            	a.trip_id,
            	a.stop_id as departure_stop_id,
            	a.stop_time as departure_time,
            	a.stop_time_unix as departure_time_unix,
            	s.stop_id as arrival_stop_id,
            	s.stop_time as arrival_time,
            	s.stop_time_unix as arrival_time_unix,
            	s.stop_time_unix - a.stop_time_unix as trip_duration
            from
            (select * from stop_events
            where data_frame_ref = '{}'
            and stop_id = {}) a
            join stop_events s
            on a.data_frame_ref = s.data_frame_ref
            and a.trip_id = s.trip_id
            and s.stop_time_unix > a.stop_time_unix""".format(data_frame_ref, dep_stop_id))

    pr.close_up_shop()


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
    raw_to_stops()
    stops_to_durations()
