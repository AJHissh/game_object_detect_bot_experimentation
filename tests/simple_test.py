import ctypes
import time

def hardware_method_1():
    """Low-level hardware simulation"""
    print("1. Testing hardware-level method 1...")
    try:
        # Use different flag combinations
        flags_combinations = [
            (0x0002, 0x0004),  # Standard
            (0x0002 | 0x0001, 0x0004 | 0x0001),  # With move
            (0x0002 | 0x8000, 0x0004 | 0x8000),  # With absolute
        ]
        
        for down_flags, up_flags in flags_combinations:
            ctypes.windll.user32.mouse_event(down_flags, 0, 0, 0, 0)
            time.sleep(0.02)
            ctypes.windll.user32.mouse_event(up_flags, 0, 0, 0, 0)
            time.sleep(0.1)
        
        print("   ✅ Hardware method 1 executed")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def hardware_method_2():
    """Extended mouse data approach"""
    print("2. Testing extended mouse data...")
    try:
        # Try with mouseData parameter
        for data in [0, 1, 100, 1000]:
            ctypes.windll.user32.mouse_event(0x0002, 0, 0, data, 0)
            time.sleep(0.01)
            ctypes.windll.user32.mouse_event(0x0004, 0, 0, data, 0)
            time.sleep(0.05)
        
        print("   ✅ Extended data method executed")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False