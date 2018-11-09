import json
import requests
import time
import sys, traceback
import boto3
import uuid
from datetime import datetime, timedelta

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

if __name__=='__main__':
    api = TransitAPI(API_KEY)

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('VehicleMonitoring')

    wait_secs = 60
    trip_thresh = 0.9
    last_query_time = datetime.now() - timedelta(seconds=wait_secs)
    last_trip_count = 0
    low_trip_count = False

    while True:
        if datetime.now() > (last_query_time + timedelta(seconds=wait_secs) or low_trip_count):
            last_query_time = datetime.now()

            response = api.get_vehicle_predictions( "SF" )
            trips = response['Siri']['ServiceDelivery']['VehicleMonitoringDelivery']['VehicleActivity']

            print('{} trips found at {}'.format(len(trips), last_query_time))

            with table.batch_writer() as batch:
                for trip in trips:
                    trip['UUID'] = str(uuid.uuid4())
                    try:
                        batch.put_item(Item=trip)
                    except Exception:
                        traceback.print_exc(file=sys.stdout)
                        continue

            if len(trips) < last_trip_count * trip_thresh:
                print('Low trip count, retrying...')
                low_trip_count = True
            else:
                low_trip_count = False
                
            last_trip_count = len(trips)

        else:
            time.sleep(1.0)
