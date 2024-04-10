from dronekit import connect, VehicleMode, LocationGlobalRelative, APIException
import time
import socket
#import exceptions
import math
import argparse

# Connect to the Vehicle
def connect_vehicle():
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    parser.add_argument('--sitl')

    args = parser.parse_args()

    connection_string = args.connect

    if not connection_string :
        import dronekit_sitl
        sitl = dronekit_sitl.start_default()
        connection_string = sitl.connection_string()
        #connection_string = '127.0.0.1:14550'

    print('Connecting to vehicle on: %s' % connection_string)
    vehicle = connect(connection_string, wait_ready=True)

    return vehicle

# Function to arm and then takeoff to a specified altitude
def arm_and_takeoff(aTargetAltitude, vehicle):
    print("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to become armable.")
        time.sleep(1)

    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)  # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        # Break and return from function just below target altitude.
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def land_vehicle(vehicle):
    prin("Now let's land")
    vehicle.mode = VehicleMode("LAND")
    vehicle.close()

if __name__ == "__main__":
    copter = connect_vehicle()
    arm_and_takeoff(copter, 10)
    print("Takeoff complete")

    # Hover for 2 seconds
    time.sleep(2)

    land_vehicle(copter)
    print("Completed")

