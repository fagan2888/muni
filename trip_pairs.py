import pandas as pd
import data_processing_planB as dp
import pickle

#Load GTFS data
df_gtfs = dp.load_gtfs_data()

df_trips = df_gtfs['trips']
df_routes = df_gtfs['routes']
df_stops = df_gtfs['stops']
df_stop_times = df_gtfs['stop_times']

df_trip_dep_stops

#Merge dataframes together to departures
df_trip_dep_stops = df_trips.merge(df_routes, on='route_id').merge(df_stop_times, on='trip_id')

#Extract hour from departure time
df_trip_dep_stops['hour'] = df_trip_dep_stops['departure_time'].str[:2].astype(int)%24

#Rename columns, drop uneeded columns
df_trip_dep_stops = df_trip_dep_stops[['route_id', 'trip_id', 'service_id', 'stop_id', 'stop_sequence', 'hour']]
df_trip_dep_stops = df_trip_dep_stops.rename(index=str, columns={"stop_id": "departure_stop_id", "stop_sequence": "departure_stop_sequence"})

#Create arrivals from departures
df_trip_arr_stops = df_trip_dep_stops.copy().rename(index=str, columns={"departure_stop_id": "arrival_stop_id", "departure_stop_sequence": "arrival_stop_sequence"})
df_trip_arr_stops = df_trip_arr_stops.drop(['hour', 'service_id'], axis=1)

#Merge together to create trip pairs
df_trip_pairs = df_trip_dep_stops.merge(df_trip_arr_stops, on =['trip_id', 'route_id'])
df_trip_pairs = df_trip_pairs[df_trip_pairs['departure_stop_sequence'] < df_trip_pairs['arrival_stop_sequence']]

df_trip_pairs['num_stops'] = df_trip_pairs['arrival_stop_sequence'] - df_trip_pairs['departure_stop_sequence']

#df_trip_pairs = df_trip_pairs.groupby(['route_id', 'service_id', 'departure_stop_id', 'hour', 'arrival_stop_id']).mean().reset_index()
#df_trip_pairs = df_trip_pairs.rename(index=str, columns={"trip_id": "trips_per_hour"})

#df_trip_pairs = pd.get_dummies(df_trip_pairs, columns=['route_short_name'])

#df_trip_pairs['num_lines'] = 1

df_trip_pairs = df_trip_pairs.groupby(['service_id', 'departure_stop_id', 'hour', 'arrival_stop_id']).mean().reset_index()
df_trip_pairs
#df_trip_pairs.to_csv('data/trip_pairs.csv', index=False)
pickle.dump(df_trip_pairs, open('df_trip_pairs.pickle', 'wb'))
