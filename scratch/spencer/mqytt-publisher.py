import paho.mqtt.client as mqtt
import time

mqttc = None
TOPIC = "is614-iot/light/sensor1"

def main() -> None:
    global mqttc

    mqttc = mqtt.Client()
    mqttc.connect("broker.mqttdashboard.com")
    mqttc.loop_start()

    while True:

        payload="light:432"
        print(f"Publish | topic: {TOPIC} | payload: {payload}")
        mqttc.publish(topic=TOPIC, payload=payload, qos=0)
        time.sleep(5)

    mqttc.loop_stop()

if __name__ == "__main__":
    main()
