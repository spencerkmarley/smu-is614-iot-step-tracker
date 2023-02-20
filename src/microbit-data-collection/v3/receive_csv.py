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

def handle_serial_data(s: serial.Serial) -> None:
    payload = s.readline().decode("utf-8").strip()
    print(payload)

# declare how long you want to collect the data      
seconds = 180 #put a shorter timing to test out first  
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
    if message[0] == 'accel': 
        # append the uuid at the front 
        accel_string = uuid + ',' + message[1] + ',' + message[2]  + ',' + message[3]
        # append the time at the front       
        current_time = time.time()
        # limit the decimal to 2 places      
        accel_string = str(round(current_time-start_time, 2)) + ',' + accel_string
        print(accel_string)
        with open('accel_' + str(filename), "a") as myfile:
            myfile.write(str(accel_string)+"\n")   

    elif message[0] == 'gyro':
        # append the uuid at the front 
        gyro_string = message[1] + ',' + message[2]  + ',' + message[3] 
        print(gyro_string)
        with open('gyro_' + str(filename), "a") as myfile:
            myfile.write(str(gyro_string)+"\n")   
            
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

######################################################## merge the csv file here: 

# accel
file_accel = open('accel_' + str(filename),"r")
# Reading file 
lines_accel = reader(file_accel)
# Converting into a list 
data_accel = list(lines_accel)

# gyro
file_gyro = open('gyro_' + str(filename),"r")
# Reading file 
lines_gyro = reader(file_gyro)
# Converting into a list 
data_gyro = list(lines_gyro)

# get the minimum number of rows due to radio package loss
min_rows = min(len(data_accel),len(data_gyro))
print(len(data_accel),len(data_gyro))

# combines the columns into a new csv file 
for row in range(min_rows): 
    new_string =  data_accel[row][0] + ',' + data_accel[row][1] + ',' \
    + data_gyro[row][0] + ',' + data_gyro[row][1] + ',' + data_gyro[row][2] + ',' \
    + data_accel[row][2] + ',' + data_accel[row][3] + ',' + data_accel[row][4]
  
    with open('combined_' + str(filename), "a") as myfile:
        myfile.write(str(new_string)+"\n")   

file_accel.close()
file_gyro.close()