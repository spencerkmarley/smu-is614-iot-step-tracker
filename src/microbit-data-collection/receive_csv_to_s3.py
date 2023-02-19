# this script will directly read off the value from the microbit via the serial port
# 0. CLOSE THE MAKECODE CONSOLE, IT WILL CAUSE DATALOSS !!! 
# 1. please change the "seconds" according to how much time you require to record 
# 2. please change the uuid accordingly 
# 3. please change the s.port accordingly, it can be COM3, COM4 etc. Need to experiment to find the right one

# To run the script simply call receive_csv.py on your python terminal 
# The file will then be saved at the same directory as this python script as a csv 
# The file is then automatically uploaded to the S3 directly using boto3
# We do not use mqtt here: 
# The reason is because each mqtt has a limit of 800 rows of data only 
# Also when uploadeding 800 rows of data, there is a 1 second delay which causes loss of incoming data stream
# Hence we use boto3 

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
import boto3
from datetime import datetime

def handle_serial_data(s: serial.Serial) -> None:
    payload = s.readline().decode("utf-8").strip()
    print(payload)

# declare how long you want to collect the data      
seconds = 30 #put a shorter timing to test out first  

# declare the uuid of the data
uuid = 'licheng_walk'+ '_' + str(datetime.now().strftime("%m-%d-%H-%M"))
    
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

# set the done bool to false first 
done = False 
# while loop 
while done == False:
    # read the raw data      
    data1 = str(s.readline().decode("utf-8").strip())
    # split the message first     
    message = data1.split(',')
    # add in the dummy data for gyro x & gyro y    
    # order of data : gx gy gz x y z 
    new_string = '0' + ',' + '0' + ',' + message[3] + ',' + message[0] + ',' + message[1] + ',' + message[2]
    
    # append the uuid at the front 
    new_string = uuid + ',' + new_string
    
    # append the time at the front       
    current_time = time.time()
    # limit the decimal to 2 places      
    new_string = str(round(current_time-start_time, 2)) + ',' + new_string
    
    print(new_string)
    if new_string is not None:
        with open(str(filename), "a") as myfile:
            myfile.write(str(new_string)+"\n")   
    
    # for tracking the end time     
    end_time = time.time()
    if end_time - start_time > seconds: 
        done = True
    
# prints the end time 
end = time.time()
print(end_time - start_time)

print('your filename is:', filename)
s3 = boto3.resource('s3')
s3.meta.client.upload_file(filename, 'smu-is614-iot-step-tracker', 'microbit/{}'.format(filename))
print('your file', filename, ' has been uploaded succesfully')

# closes the port 
s.close()
print('port closed')