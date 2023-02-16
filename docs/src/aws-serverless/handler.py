import AWSIoTPythonSDK.MQTTLib as AWSIoTPyMQTT
import json
import time

CLIENT = "microbit"
ENDPOINT = "a336e03nfdcvqi-ats.iot.ap-southeast-1.amazonaws.com"
ROOT_PEM = "certificates/root.pem"
PRIVATE_PEM_KEY = "certificates/private.pem.key"
CERTIFICATE_PEM_CRT = "certificates/certificate.pem.crt"
TOPIC = "microbit"

def handler(event, context):
    
    data = {
    "timestamp": int(round(time.time() * 1000)),
    "id": event["id"],
    "gyro_x": event["gyro_x"],
    "gyro_y": event["gyro_y"],
    "gyro_z": event["gyro_z"],
    "accel_x": event["accel_x"],
    "accel_y": event["accel_y"],
    "accel_z": event["accel_z"],
    }
    json_string = json.dumps(data)

    return {
        'statusCode': 200,
        'body': json_string
    }