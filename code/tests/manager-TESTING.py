#!/usr/bin/python3

import psutil


# horrible hack just to keep this reference working on MY (Albert) laptop
class machine_config:
    machine_temp_reference = "thinkpad"

temp = psutil.sensors_temperatures()

# Ideally would be fetched from system, but no luck on my machine, can be coded to call if there or fall back on a preset value
critical = 84.85

def fetch_temps():
    print("")
    while(True):
        cputemp = temp[machine_config.machine_temp_reference][0].current
        if cputemp > critical:
            print(f"\033[A \033[  CCPU temp: {cputemp}")

def main():
   print("")
   fetch_temps()

if __name__ == "__main__":
    main()