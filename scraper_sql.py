import json
import requests
import time
import sys, traceback
import boto3
import uuid
from datetime import datetime, timedelta
import psycopg2
import csv

API_KEY = "691b2ac5-a085-4194-b877-2997a4fac382"

class TransitAPI(object):
    def __init__(self, key):
        self.key = key

    def get_stop_predictions(self, operator):
        response = requests.get('http://api.511.org/transit/StopMonitoring?api_key=%s&agency=%s'%(self.key,operator))
        return json.loads( response.content )

    def get_vehicle_predictions(self, operator):
        response = requests.get('http://api.511.org/transit/VehicleMonitoring?api_key=%s&agency=%s'%(self.key,operator))
        return json.loads( response.content )

    def get_operators(self):
        response = requests.get('http://api.511.org/transit/operators?api_key=%s&Format=json'%(self.key))
        return json.loads( response.content )

def trip_parser(raw_trip):
    line_ref = trip['MonitoredVehicleJourney']['LineRef']
    recorded_time = trip['RecordedAtTime'].replace('T', ' ').replace('Z', '')
    valid_until_time = trip['ValidUntilTime'].replace('T', ' ').replace('Z', '')
    direction_ref = trip['MonitoredVehicleJourney']['DirectionRef']
    data_frame_ref = trip['MonitoredVehicleJourney']['FramedVehicleJourneyRef']['DataFrameRef']
    journey_ref = trip['MonitoredVehicleJourney']['FramedVehicleJourneyRef']['DatedVehicleJourneyRef']
    line_name = trip['MonitoredVehicleJourney']['PublishedLineName']
    operator_ref = trip['MonitoredVehicleJourney']['OperatorRef']
    monitored = trip['MonitoredVehicleJourney']['Monitored']
    vehicle_lat = trip['MonitoredVehicleJourney']['VehicleLocation']['Longitude']
    vehicle_lon = trip['MonitoredVehicleJourney']['VehicleLocation']['Latitude']
    vehicle_ref = trip['MonitoredVehicleJourney']['VehicleRef']
    stop_point_ref = trip['MonitoredVehicleJourney']['MonitoredCall']['StopPointRef']
    visit_num = trip['MonitoredVehicleJourney']['MonitoredCall']['VisitNumber']
    stop_point_name = trip['MonitoredVehicleJourney']['MonitoredCall']['StopPointName']
    expected_arrival_time = trip['MonitoredVehicleJourney']['MonitoredCall']['ExpectedArrivalTime'].replace('T', ' ').replace('Z', '')
    expected_departure_time = trip['MonitoredVehicleJourney']['MonitoredCall']['ExpectedDepartureTime'].replace('T', ' ').replace('Z', '')
    parsed_trip = [line_ref,
                   recorded_time,
                   valid_until_time,
                   direction_ref,
                   data_frame_ref,
                   journey_ref,
                   line_name,
                   operator_ref,
                   monitored,
                   vehicle_lat,
                   vehicle_lon,
                   vehicle_ref,
                   stop_point_ref,
                   visit_num,
                   stop_point_name,
                   expected_arrival_time,
                   expected_departure_time]
    return parsed_trip


if __name__=='__main__':
    with open('credentials.json') as json_data:
        credentials = json.load(json_data)

    api = TransitAPI(API_KEY)

    s3 = boto3.resource('s3')
    conn = psycopg2.connect('host=jonobate.c9xvjgh0xspr.us-east-1.redshift.amazonaws.com port=5439 dbname=muni user={} password={}'.format(credentials['user'], credentials['password']))

    # Open a cursor to perform database operations
    cur = conn.cursor()

    wait_secs = 10
    trip_thresh = 0.9
    last_query_time = datetime.now() - timedelta(seconds=wait_secs)
    last_trip_count = 0
    low_trip_count = False

    while True:
        if datetime.now() > (last_query_time + timedelta(seconds=wait_secs) or low_trip_count):
            try:
                last_query_time = datetime.now()

                response = api.get_vehicle_predictions( "SF" )
                trips = response['Siri']['ServiceDelivery']['VehicleMonitoringDelivery']['VehicleActivity']

                print('{} trips found at {}'.format(len(trips), last_query_time))

                results = []
                for trip in trips:
                    results.append(trip_parser(trip))
                with open("output.csv", "w") as f:
                    writer = csv.writer(f, delimiter=',', quotechar='"')
                    writer.writerows(results)

                s3.meta.client.upload_file('output.csv', 'jonobate-bucket', 'output.csv')

                # Execute a command
                cur.execute('''copy vehicle_monitoring
                from 's3://jonobate-bucket/output.csv'
                iam_role 'arn:aws:iam::614550856824:role/AWS-admin'
                delimiter ',' ''')

                # Make the changes to the database persistent
                conn.commit()

                if len(trips) < last_trip_count * trip_thresh:
                    print('Low trip count, retrying...')
                    low_trip_count = True
                else:
                    low_trip_count = False

                last_trip_count = len(trips)
            except Exception as e:
                print(e)
                print('Something failed, retrying...')
                time.sleep(1.0)
                continue
        else:
            time.sleep(1.0)
