from piezo_control import PiezoController, PiezoError

def main():
    print("--- Thorlabs Piezo Controller Demo ---")
    
    try:
        # 1. List connected devices
        print("Searching for connected devices...")
        devices = PiezoController.list_devices()
        if not devices:
            print("No devices found. Please check the connection.")
            return
        
        print(f"Found {len(devices)} device(s):")
        for i, (sn, dtype) in enumerate(devices):
            print(f"[{i}] Serial: {sn}, Type: {dtype}")
        
        # Use the first found device for the demo
        target_sn, target_type = devices[0]
        print(f"\nUsing device: {target_sn} ({target_type})")
        
        # 2. Initialize and open controller
        # Using context manager for automatic closing
        with PiezoController(target_sn) as piezo:
            print("Opening device...")
            piezo.open()
            print("Device opened successfully.")
            
            # 3. Get device ID
            device_id = piezo.get_id()
            print(f"Device ID: {device_id}")
            
            # 4. Voltage Control Demo
            # Get current voltage for X axis
            current_v = piezo.get_voltage('X')
            print(f"Current X voltage: {current_v}V")
            
            # Set a new voltage (Caution: Ensure this is safe for your specific setup!)
            target_v = 10.0 
            print(f"Setting X voltage to {target_v}V...")
            piezo.set_voltage('X', target_v)
            
            # Verify the change
            new_v = piezo.get_voltage('X')
            print(f"Verified X voltage: {new_v}V")
            
            # 5. Test XYZ simultaneous set
            print("Setting XYZ voltages to (5, 5, 5)...")
            piezo.set_xyz_voltages(5.0, 5.0, 5.0)
            print(f"XYZ Voltages: {piezo.get_xyz_voltages()}")
            
            # Reset X back to original if we recorded it
            print(f"Resetting X back to {current_v}V...")
            piezo.set_voltage('X', current_v)

    except PiezoError as e:
        print(f"Piezo Control Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
