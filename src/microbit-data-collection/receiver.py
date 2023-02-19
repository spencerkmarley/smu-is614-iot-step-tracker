def on_received_string(receivedString):
    global count, received_val
    count += 1
    received_val = receivedString
    serial.write_line(received_val)
radio.on_received_string(on_received_string)

received_val = ""
count_old = 0
count = 0
radio.set_group(1)

def on_forever():
    global count_old
    if count_old == count:
        basic.show_icon(IconNames.NO)
    else:
        basic.show_icon(IconNames.YES)
    count_old = count
    basic.pause(100)
basic.forever(on_forever)
