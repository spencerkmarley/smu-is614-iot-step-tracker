import dweepy
import pprint

def main() -> None:
    thing_name = "is614-iot/light/sensor1"
    response = dweepy.dweet_for(thing_name, {'x': '1', 'y' : '2', 'z' : '3'})

    print(f"dweet's response:")
    pprint.pprint(response)

if __name__ == "__main__":
    main()
