def on_button_pressed_a():
    global sendData
    sendData = not (sendData)
input.on_button_pressed(Button.A, on_button_pressed_a)

gyro_z_val = 0
gyro_y_val = 0
gyro_x_val = 0
z_val = 0
y_val = 0
x_val = 0
gyro = ""
accel = ""
sendData = False
SENMPU6050.init_mpu6050()
radio.set_transmit_power(7)
radio.set_group(1)
duration = 1000
sendData = False

def on_forever():
    global accel, gyro
    while sendData:
        accel = "accel" + "," + ("" + str(Math.round(input.acceleration(Dimension.X) / 10))) + "," + ("" + str(Math.round(input.acceleration(Dimension.Y) / 10))) + "," + ("" + str(Math.round(input.acceleration(Dimension.Z) / 10)))
        gyro = "gyro" + "," + ("" + str(Math.round(SENMPU6050.gyroscope(axisXYZ.X, gyroSen.RANGE_250_DPS) / 10))) + "," + ("" + str(Math.round(SENMPU6050.gyroscope(axisXYZ.Y, gyroSen.RANGE_250_DPS) / 10))) + "," + ("" + str(Math.round(SENMPU6050.gyroscope(axisXYZ.Z, gyroSen.RANGE_250_DPS) / 10)))
        # serial.write_value("X", input.acceleration(Dimension.X))
        # serial.write_value("Y", input.acceleration(Dimension.Y))
        # serial.write_value("Z", input.acceleration(Dimension.Z))
        radio.send_string(accel)
        # serial.write_value("g_X",
        # Math.round(SENMPU6050.gyroscope(axisXYZ.X, gyroSen.RANGE_250_DPS)))
        # serial.write_value("g_Y",
        # Math.round(SENMPU6050.gyroscope(axisXYZ.Y, gyroSen.RANGE_250_DPS)))
        # serial.write_value("g_Z",
        # Math.round(SENMPU6050.gyroscope(axisXYZ.Z, gyroSen.RANGE_250_DPS)))
        # radio.send_string(gyro)
        basic.pause(20)
        # serial.write_value("X", input.acceleration(Dimension.X))
        # serial.write_value("Y", input.acceleration(Dimension.Y))
        # serial.write_value("Z", input.acceleration(Dimension.Z))
        radio.send_string(gyro)
        # serial.write_value("g_X",
        # Math.round(SENMPU6050.gyroscope(axisXYZ.X, gyroSen.RANGE_250_DPS)))
        # serial.write_value("g_Y",
        # Math.round(SENMPU6050.gyroscope(axisXYZ.Y, gyroSen.RANGE_250_DPS)))
        # serial.write_value("g_Z",
        # Math.round(SENMPU6050.gyroscope(axisXYZ.Z, gyroSen.RANGE_250_DPS)))
        # radio.send_string(gyro)
        basic.pause(20)
basic.forever(on_forever)

def on_forever2():
    if sendData == True:
        basic.show_icon(IconNames.HAPPY)
    else:
        basic.show_icon(IconNames.ASLEEP)
basic.forever(on_forever2)

def on_forever3():
    global x_val, y_val, z_val, gyro_x_val, gyro_y_val, gyro_z_val
    x_val = Math.round(input.acceleration(Dimension.X) / 1)
    y_val = Math.round(input.acceleration(Dimension.Y) / 1)
    z_val = Math.round(input.acceleration(Dimension.Z) / 1)
    gyro_x_val = Math.round(SENMPU6050.gyroscope(axisXYZ.X, gyroSen.RANGE_250_DPS) / 1)
    gyro_y_val = Math.round(SENMPU6050.gyroscope(axisXYZ.Y, gyroSen.RANGE_250_DPS) / 1)
    gyro_z_val = Math.round(SENMPU6050.gyroscope(axisXYZ.Z, gyroSen.RANGE_250_DPS) / 1)
    serial.write_value("x", x_val)
    serial.write_value("Y", y_val)
    serial.write_value("Z", z_val)
    serial.write_value("gyro_x", gyro_x_val)
    serial.write_value("gyro_y", gyro_y_val)
    serial.write_value("gyro_z", gyro_z_val)
    basic.pause(50)
basic.forever(on_forever3)
