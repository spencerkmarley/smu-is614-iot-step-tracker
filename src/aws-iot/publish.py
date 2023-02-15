import AWSIoTPythonSDK.MQTTLib as AWSIoTPyMQTT
import json
import time

CLIENT = "microbit"
ENDPOINT = "a336e03nfdcvqi-ats.iot.ap-southeast-1.amazonaws.com"
ROOT_PEM = "certificates/root.pem"
PRIVATE_PEM_KEY = "certificates/private.pem.key"
CERTIFICATE_PEM_CRT = "certificates/certificate.pem.crt"
TOPIC = "microbit"

ID = 123456
GYRO_X = 10
GYRO_Y = 84.76812997658814
GYRO_Z = -7.0005309400135305
ACCEL_X = -48.7129175729368
ACCEL_Y = -52.266628054100394
ACCEL_Z = 27.931961024295653

data = {
    "timestamp": int(round(time.time() * 1000)),
    "id": ID,
    "gyro_x": GYRO_X,
    "gyro_y": GYRO_Y,
    "gyro_z": GYRO_Z,
    "accel_x": ACCEL_X,
    "accel_y": ACCEL_Y,
    "accel_z": ACCEL_Z
}
json_string = json.dumps(data)

def publish(client, endpoint, root_pem, private_pem_key, certificate_pem_crt, topic, payload):
    
    myAWSIoTMQTTClient = AWSIoTPyMQTT.AWSIoTMQTTClient(client)
    myAWSIoTMQTTClient.configureEndpoint(endpoint, 8883)
    myAWSIoTMQTTClient.configureCredentials(root_pem, private_pem_key, certificate_pem_crt)

    myAWSIoTMQTTClient.connect()
    myAWSIoTMQTTClient.publish(topic, payload, 2)
    myAWSIoTMQTTClient.disconnect()

publish(CLIENT, ENDPOINT, ROOT_PEM, PRIVATE_PEM_KEY, CERTIFICATE_PEM_CRT, TOPIC, json_string)