import paho.mqtt.client as mqtt
import time

mqttc = None

TOPIC = "is614-iot/light/sensor1"

def handle_mqtt_connack(client, userdata, flags, rc) -> None:
    print(f"MQTT broker said: {mqtt.connack_string(rc)}")
    if rc == 0:
        client.is_connected = True

    client.subscribe(f"{TOPIC}")
    print(f"Subscribed to: {TOPIC}")
    print(f"Publish something to {TOPIC} and the messages will appear here.")

def handle_mqtt_message(client, userdata, msg) -> None:
    print(f"received msg | topic: {msg.topic} | payload: {msg.payload.decode('utf8')}")

def main() -> None:
    global mqttc

    mqttc = mqtt.Client()
    mqttc.on_connect = handle_mqtt_connack
    mqttc.on_message = handle_mqtt_message
    mqttc.is_connected = False
    mqttc.connect("broker.mqttdashboard.com")
    mqttc.loop_start()
    time_to_wait_secs = 5
    waited_for_too_long = False
    while not mqttc.is_connected and not waited_for_too_long:
        time.sleep(0.1)
        time_to_wait_secs -= 0.1
        if time_to_wait_secs <= 0:
            waited_for_too_long = True

    if waited_for_too_long:
        print(f"Can't connect to broker.mqttdashboard.com, waited for too long")
        return

    while True:
        time.sleep(10)

    mqttc.loop_stop()

if __name__ == "__main__":
    main()