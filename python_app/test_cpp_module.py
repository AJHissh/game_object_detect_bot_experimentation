import sys
import os
import cv2
import numpy as np

# Add the C++ module to path
cpp_build_path = os.path.join(os.path.dirname(__file__), '..', 'cpp_module', 'build', 'Release', 'Release')
sys.path.append(cpp_build_path)

try:
    import bot_controller
    print("✅ C++ module imported successfully!")
    
    # Test the controller
    controller = bot_controller.BotController()
    if controller.initialize():
        print("✅ Controller initialized successfully!")
        
        # Get screen resolution
        width, height = controller.get_screen_resolution()
        print(f"✅ Screen resolution: {width}x{height}")
        
        # Test mouse movement
        print("Testing mouse movement to (100, 100)...")
        if controller.move_mouse(100, 100):
            print("✅ Mouse movement test passed!")
        else:
            print("❌ Mouse movement test failed!")
        
        # Test screen capture
        print("Testing screen capture...")
        frame = controller.capture_region(0, 0, 800, 600)
        if frame is not None and frame.size > 0:
            print(f"✅ Screen capture test passed! Frame shape: {frame.shape}")
            
            # Save test image to verify
            cv2.imwrite('test_capture.jpg', frame)
            print("✅ Test image saved as 'test_capture.jpg'")
        else:
            print("❌ Screen capture test failed!")
        
        # Test click
        print("Testing mouse click...")
        if controller.click():
            print("✅ Mouse click test passed!")
        else:
            print("❌ Mouse click test failed!")
            
    else:
        print("❌ Failed to initialize controller")
        
except ImportError as e:
    print(f"❌ Failed to import module: {e}")
    print("Make sure the .pyd file is in the correct directory")
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()