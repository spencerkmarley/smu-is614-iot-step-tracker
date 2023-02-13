import requests
import dweepy
import pprint

def main() -> None:
    thing_name = "is614-iot/light/sensor1"

    while True:
        try:
            for dweet in dweepy.listen_for_dweets_from(thing_name):
                print(f"dweet = {dweet}")
        except requests.exceptions.ConnectionError as e:
            print(f"dweet.io closed the connection, reconnecting")

if __name__ == "__main__":
    main()
