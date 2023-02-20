# this script will directly read off the value from the microbit via the serial port 
# 0. CLOSE THE MAKECODE CONSOLE, IT WILL CAUSE DATALOSS !!! 
# 1. please change the "seconds" according to how much time you require to record 
# 2. please change the uuid accordingly 
# 3. please change the s.port accordingly, it can be COM3, COM4 etc. Need to experiment to find the right one

# To run the script simply call receive_csv.py on your python terminal 
# The file will then be saved at the same directory as this python script as a csv 
# From there, you may then upload the csv file to the S3 directly without mqtt for convenience
# This method is same as mqtt, just that we upload the data directly instead of mqtt
# The reason is because each mqtt has a limit of 800 rows of data only 

# To record the data: 
# Load the code to the receiver microbit.
# Load the code to the transmitter microbit.
# Run the receive_csv.py on python terminal, the script will then standby.
# Press button A on the transmitter microbit, the face will then change to smiley face, transmit will start.
# On the receiving microbit, the cross icon will become tick icon to indicate that radio message has been received.
# On the python terminal, you will see entries start being recorded. 
# After the number of seconds x has elasped, the csv will then be saved. 
# Press button A on the transmitter microbit to stop the transmit. 

import serial
import time
from datetime import datetime
from csv import reader
import os
import AWSIoTPythonSDK.MQTTLib as AWSIoTPyMQTT

CLIENT = "microbit"
ENDPOINT = "a336e03nfdcvqi-ats.iot.ap-southeast-1.amazonaws.com"
# download the certificates from our common folder 
ROOT_PEM = "certificates/root.pem"
PRIVATE_PEM_KEY = "certificates/33d027bba80745d5587a65c0d230a4de5eb7c3917c49d85e2d456d7e83670b56-private.pem.key"
CERTIFICATE_PEM_CRT = "certificates/33d027bba80745d5587a65c0d230a4de5eb7c3917c49d85e2d456d7e83670b56-certificate.pem.crt"
TOPIC = "microbit"

def publish(client, endpoint, root_pem, private_pem_key, certificate_pem_crt, topic, payload):
    
    myAWSIoTMQTTClient = AWSIoTPyMQTT.AWSIoTMQTTClient(client)
    myAWSIoTMQTTClient.configureEndpoint(endpoint, 8883)
    myAWSIoTMQTTClient.configureCredentials(root_pem, private_pem_key, certificate_pem_crt)

    myAWSIoTMQTTClient.connect()
    myAWSIoTMQTTClient.publish(topic, payload, 1)
    myAWSIoTMQTTClient.disconnect()

def handle_serial_data(s: serial.Serial) -> None:
    payload = s.readline().decode("utf-8").strip()
    print(payload)

# declare how long you want to collect the data      
seconds = 240 #put a shorter timing to test out first  
# declare the uuid of the data
# uuid = 'licheng_walk'+ '_' + str(datetime.now().strftime("%m-%d-%H-%M"))
uuid = 'licheng_walk'
    
s = serial.Serial()
s.baudrate = 115200
s.port = "COM4"
s.set_buffer_size(rx_size = 12800, tx_size = 12800)
s.open()
print('port opened')
s.reset_input_buffer()
s.reset_output_buffer()

# print the current time 
print(datetime.now())
# create the filenanme 
filename = "datadump" + str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + ".csv"

# records the start time so we can track the time elasped
start_time = time.time()

queue_accel = []
queue_gyro = [] 
payload_count = 0 
string_collection = ''
# set the done bool to false first 
done = False 
# while loop 
while done == False:
    # read the raw data   

    data1 = str(s.readline().decode("utf-8").strip())
    
    # split the message first     
    message = data1.split(',')
    
    # check if it is accel message      
    if message[0] == 'accel': 
        # get current time       
        current_time = time.time()        
        accel_entry = [str(round(current_time-start_time, 2)), uuid, message[1], message[2], message[3]]
        # send to queue
        queue_accel.append(accel_entry)
        
    # check if it is gyro message 
    elif message[0] == 'gyro':
        gyro_entry = [message[1], message[2], message[3]]
        # send to queue
        queue_gyro.append(gyro_entry)
            
    # check if both queue are not empty: 
    if (len(queue_accel) !=0 and len(queue_gyro) !=0): 
        a = queue_accel.pop(0)
        g = queue_gyro.pop(0)
        string_final = a[0] + ',' + a[1] + ',' + g[0] + ',' + g[1] + ',' + g[2] + ',' + a[2] + ',' + a[3] + ',' + a[4]
        
        # create/update payload
        
        # within x number of rows, time to add a message 
        if payload_count < 200:
            print(string_final)
            string_collection = string_collection + '\n' + string_final
            payload_count += 1
        # exceed x number of rows, time to send message 
        else: 
#             print(string_collection)
            publish(CLIENT, ENDPOINT, ROOT_PEM, PRIVATE_PEM_KEY, CERTIFICATE_PEM_CRT, TOPIC, string_collection)
            string_collection = '' 
            payload_count = 0 
            
    # for tracking the end time     
    end_time = time.time()
    if end_time - start_time > seconds: 
        done = True
    
# prints the end time 
end = time.time()
print(end_time - start_time)
# closes the port 
s.close()
print('port closed')
print(len(queue_accel), len(queue_gyro))

