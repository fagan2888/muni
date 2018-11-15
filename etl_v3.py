import json
import numpy as np
import pandas as pd
import pandas_redshift as pr
import scipy.stats as st

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
            	s.stop_time_unix - a.stop_time_unix as trip_duration,
                date_trunc('hour', a.stop_time) as departure_time_hour
            from
            (select * from stop_events
            where data_frame_ref = '{}'
            and stop_id = {}) a
            join stop_events s
            on a.data_frame_ref = s.data_frame_ref
            and a.trip_id = s.trip_id
            and s.stop_time_unix > a.stop_time_unix""".format(data_frame_ref, dep_stop_id))

    pr.close_up_shop()



def durs_to_dists():

    connect_to_redshift()
    connect_to_s3()

    #Note: this processes data not already in distributions. Assumes we do one hour at a time, no subdividing of hours.
    df = pr.redshift_to_pandas("""select a.* from
        (select data_frame_ref, departure_time_hour from trip_durations group by data_frame_ref, departure_time_hour) a
        left join
        (select data_frame_ref, departure_time_hour from distributions group by data_frame_ref, departure_time_hour) b
        on a.data_frame_ref = b.data_frame_ref
        	and a.departure_time_hour = b.departure_time_hour
        where b.data_frame_ref is null
        	and b.departure_time_hour is null
            and a.data_frame_ref < trunc(convert_timezone('US/Pacific', GETDATE()))
            order by a.data_frame_ref, a.departure_time_hour;""")

    #Randomize order, so we can get some samples from everywhere...
    df = df.sample(frac=1).reset_index(drop=True)

    n_days_hours = df.shape[0]

    #For each day and departure stop:
    for i, row in df.iterrows():
        data_frame_ref = row['data_frame_ref']
        departure_time_hour = row['departure_time_hour']
        print("Processing data_frame_ref {}, departure_time_hour {} ({} of {})".format(data_frame_ref, departure_time_hour, (i+1), n_days_hours))

        #Calculate base timestamps for this day
        minutes = pd.DataFrame(np.arange(0,60), columns=['minute'])
        minutes['key'] = 0

        df_hour = pr.redshift_to_pandas("""select *,
                                            date_trunc('min', departure_time) as departure_time_minute
                                            from trip_durations
                                            where data_frame_ref = '{}'
                                            and departure_time_hour = '{}' """.format(data_frame_ref, departure_time_hour))



        results = []

        n_dep_stops = len(df_hour['departure_stop_id'].unique())

        #For each arrival stop:
        for j, departure_stop_id in enumerate(df_hour['departure_stop_id'].unique()):
            print("Processing departure_stop_id {} ({} of {})".format(departure_stop_id, (j+1), n_dep_stops))

            #For each departure stop:
            for k, arrival_stop_id in enumerate(df_hour[df_hour['departure_stop_id']==departure_stop_id]['arrival_stop_id'].unique()):

                #Select data
                df_dist = df_hour[(df_hour['departure_stop_id']==departure_stop_id) & (df_hour['arrival_stop_id']==arrival_stop_id)]

                #Create date array
                date = pd.DataFrame([departure_time_hour], columns=['departure_time_hour'])
                date['key'] = 0

                #Create base array
                base = date.merge(minutes)
                base['departure_time_minute'] = base['departure_time_hour'] + pd.to_timedelta(base.minute, unit='m')
                base = base[['departure_time_minute']]
                base['departure_time_minute_unix'] = (base['departure_time_minute'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

                df_dist = base.merge(df_dist, on='departure_time_minute', how='left')
                df_dist = df_dist.fillna(method='bfill')
                df_dist['total_journey_time'] = df_dist['arrival_time_unix'] - df_dist['departure_time_minute_unix']
                df_dist = df_dist.dropna(subset=['total_journey_time'])

                data = df_dist['total_journey_time']

                try:
                    # fit dist to data
                    params = st.gengamma.fit(data)

                    y, x = np.histogram(data)
                    x = (x + np.roll(x, -1))[:-1] / 2.0

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = st.gengamma.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    results.append([data_frame_ref, departure_time_hour, departure_stop_id, arrival_stop_id, arg[0], arg[1], loc, scale, sse])
                except Exception as e:
                    print(e)
                    continue
        #Only bother with this if we actually have stop events...
        if len(results) == 0:
            print("No distributions for data_frame_ref {}, departure_time_hour {}, skipping...".format(data_frame_ref, departure_time_hour))
        else:
            print ("Writing distributions to Redshift...")
            df_results = pd.DataFrame(results, columns=['data_frame_ref', 'departure_time_hour', 'departure_stop_id', 'arrival_stop_id', 'shape_a', 'shape_c', 'loc', 'scale', 'sse'])
            pr.pandas_to_redshift(data_frame = df_results,
                                redshift_table_name = 'distributions',
                                append = True)

    pr.close_up_shop()



'''
def stops_to_dists():

    connect_to_redshift()
    connect_to_s3()

    #Note: this processes all data
    df = pr.redshift_to_pandas("""select data_frame_ref, stop_id
            from stop_events
            where data_frame_ref < trunc(convert_timezone('US/Pacific', GETDATE()))
            group by data_frame_ref, stop_id
            order by data_frame_ref, stop_id;""")

    n_days_dep_stops = df.shape[0]

    data_frame_ref = None

    #For each day and departure stop:
    for i, row in df.iterrows():
        #If it's a new day, or first run...
        if data_frame_ref !=row['data_frame_ref']:
            data_frame_ref = row['data_frame_ref']
            #Calculate base timestamps for this day
            minutes = pd.DataFrame(np.arange(0,60), columns=['minute'])
            minutes['key'] = 0
            hours = pd.DataFrame(np.arange(0,24), columns=['hour'])
            hours['key'] = 0
            date = pd.DataFrame([pd.to_datetime(data_frame_ref)], columns=['local_date'])
            date['key'] = 0
            base = date.merge(hours).merge(minutes)
            base['local_departure_minute'] = base['local_date'] + pd.to_timedelta(base.hour, unit='h') + pd.to_timedelta(base.minute, unit='m')
            base = base[['local_departure_minute']]
            base['local_departure_minute_unix'] = (base['local_departure_minute'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        dep_stop_id = row['stop_id']

        print("Processing data_frame_ref {}, departure_stop_id {} ({} of {})".format(data_frame_ref, dep_stop_id, (i+1), n_days_dep_stops))

        #Get trip lengths
        df = pr.redshift_to_pandas("""select a.data_frame_ref,
            	a.trip_id,
            	a.stop_id as departure_stop_id,
            	a.stop_time as departure_time,
            	a.stop_time_unix as departure_time_unix,
            	s.stop_id as arrival_stop_id,
            	s.stop_time as arrival_time,
            	s.stop_time_unix as arrival_time_unix,
                convert_timezone('US/Pacific', date_trunc('hour', a.stop_time)) as local_departure_hour,
                convert_timezone('US/Pacific', date_trunc('min', a.stop_time)) as local_departure_minute,
                convert_timezone('US/Pacific', s.stop_time) as local_arrival_time
            from
            (select * from stop_events
            where data_frame_ref = '{}'
            and stop_id = {}) a
            join stop_events s
            on a.data_frame_ref = s.data_frame_ref
            and a.trip_id = s.trip_id
            and s.stop_time_unix > a.stop_time_unix""".format(data_frame_ref, dep_stop_id))

        results = []
        #For each arrival stop:
        for i, arrival_stop_id in enumerate(df['arrival_stop_id'].unique()):
            #print('Stop: {}'.format(arrival_stop_id))
            df_arr_stop = df[df['arrival_stop_id']==arrival_stop_id]
            df_arr_stop = base.merge(df_arr_stop, on='local_departure_minute', how='left')
            df_arr_stop = df_arr_stop.fillna(method='bfill')
            df_arr_stop['local_arrival_time_unix'] = (df_arr_stop['local_arrival_time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            df_arr_stop['total_journey_time'] = df_arr_stop['local_arrival_time_unix'] - df_arr_stop['local_departure_minute_unix']
            df_arr_stop = df_arr_stop.dropna(subset=['total_journey_time'])
            for j, local_hour in enumerate(df['local_departure_hour'].unique()):
                #print(' Hour: {}'.format(local_hour))
                data = df_arr_stop[df_arr_stop['local_departure_hour']==local_hour]['total_journey_time']

                try:
                    # fit dist to data
                    params = st.gengamma.fit(data)

                    y, x = np.histogram(data)
                    x = (x + np.roll(x, -1))[:-1] / 2.0

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = st.gengamma.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    results.append([trip_id, local_hour, dep_stop_id, arrival_stop_id, arg[0], arg[1], loc, scale, sse])
                except:
                    continue
        #Only bother with this if we actually have stop events...
        if len(results) == 0:
            print("No distributions for data_frame_ref {}, departure_stop_id {}, skipping...".format(data_frame_ref, dep_stop_id))
        else:
            print ("Writing distributions to Redshift...")
            df_results = pd.DataFrame(results, columns=['trip_id', 'local_hour_timestamp', 'departure_stop_id', 'arrival_stop_id', 'shape1', 'shape2', 'loc', 'scale', 'sse'])
            pr.pandas_to_redshift(data_frame = df_results,
                                redshift_table_name = 'distributions',
                                append = True)

    pr.close_up_shop()

'''

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
    durs_to_dists()
