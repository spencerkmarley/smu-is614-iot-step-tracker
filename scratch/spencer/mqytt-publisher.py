import paho.mqtt.client as mqtt
import time

# MQTT client object
mqttc = None

# Topic to publish to.
TOPIC = "smu-is614-iot-step-tracker"

# Handles an MQTT client connect event
# This function is called once, just after the mqtt client is connected to the server.
def handle_mqtt_connack(client, userdata, flags, rc) -> None:
    print(f"MQTT broker said: {mqtt.connack_string(rc)}")
    if rc == 0:
        client.is_connected = True

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(f"{TOPIC}")
    print(f"Subscribed to: {TOPIC}")
    print(f"Publish something to {TOPIC} and the messages will appear here.")

# Handles an incoming message from the MQTT broker.
def handle_mqtt_message(client, userdata, msg) -> None:
    print(f"received msg | topic: {msg.topic} | payload: {msg.payload.decode('utf8')}")

def main() -> None:
    global mqttc

    # Create mqtt client
    mqttc = mqtt.Client()

    # Register callbacks
    mqttc.on_connect = handle_mqtt_connack
    mqttc.on_message = handle_mqtt_message

    # Set this flag to false first, handle_mqtt_connack will set it to true later
    mqttc.is_connected = False

    # Connect to broker
    mqttc.connect("broker.mqttdashboard.com")

    # start the mqtt client loop
    mqttc.loop_start()

    # approximate amount of time to wait for client to be connected
    time_to_wait_secs = 5

    # keep looping until either the client is connected, or waited for too long
    waited_for_too_long = False
    while not mqttc.is_connected and not waited_for_too_long:
        # sleep for 0.1s
        time.sleep(0.1)
        time_to_wait_secs -= 0.1

        # set this to true if waited for too long
        if time_to_wait_secs <= 0:
            waited_for_too_long = True

    # exit if client couldn't connect even after waiting for a long time
    if waited_for_too_long:
        
        logger.error(f"Can't connect to broker.mqttdashboard.com, waited for too long")
        return

    # Loopy loop
    while True:

        # Data that will be published
        payload="iot a toi"

        # Print some debugging info
        print(f"Publish | topic: {TOPIC} | payload: {payload}")

        # Publish data to MQTT broker
        mqttc.publish(topic=TOPIC, payload=payload, qos=0)

        # Wait for some time before publishing again, don't spam
        time.sleep(5)

    # Stop the MQTT client
    mqttc.loop_stop()


if __name__ == "__main__":
    main()
