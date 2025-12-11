import sys
import os
import cv2
import numpy as np
import pyautogui
import time
import ctypes

# Hardware-level mouse control
user32 = ctypes.windll.user32

# Add the C++ module to path
build_path = os.path.join(os.path.dirname(__file__), '..', 'cpp_module', 'build', 'Release', 'Release')
sys.path.append(build_path)

try:
    import bot_controller
    CPP_AVAILABLE = True
    print("✅ C++ module loaded successfully! (Screen capture optimized)")
except ImportError as e:
    print(f"❌ C++ module not available: {e}")
    CPP_AVAILABLE = False

class CppController:
    def __init__(self):
        self.controller = None
        self.screen_width = user32.GetSystemMetrics(0)  # Use your exact method
        self.screen_height = user32.GetSystemMetrics(1) # Use your exact method
        
        if CPP_AVAILABLE:
            try:
                self.controller = bot_controller.BotController()
                if self.controller.initialize():
                    print(f"✅ Screen resolution: {self.screen_width}x{self.screen_height}")
                else:
                    print("❌ Failed to initialize C++ controller")
                    self.controller = None
            except Exception as e:
                print(f"❌ Error initializing C++ controller: {e}")
                self.controller = None
        else:
            print(f"✅ Screen resolution (fallback): {self.screen_width}x{self.screen_height}")
    
    def is_available(self):
        return self.controller is not None
    
    def capture_region(self, x, y, width, height):
        """Use C++ for fast screen capture, fallback to Python if needed"""
        if self.is_available():
            try:
                frame = self.controller.capture_region(x, y, width, height)
                if frame is not None and frame.size > 0:
                    return np.array(frame, dtype=np.uint8)
            except Exception as e:
                print(f"⚠️ C++ capture failed, using Python fallback: {e}")
        
        # Python fallback (your original method)
        try:
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            img_np = np.array(screenshot)
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"❌ Python capture also failed: {e}")
            return None
    
    def move_mouse_direct_input(self, x, y):
        """EXACT COPY of your working mouse movement function"""
        try:
            normalized_x = int((x / self.screen_width) * 65535)
            normalized_y = int((y / self.screen_height) * 65535)
            user32.mouse_event(0x8001, normalized_x, normalized_y, 0, 0)
            return True
        except Exception as e:
            print(f"❌ Hardware mouse movement failed: {e}")
            return False
    
    def click_hardware(self):
        """EXACT COPY of your working click function"""
        try:
            user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
            time.sleep(0.05)
            user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
            return True
        except Exception as e:
            print(f"❌ Hardware click failed: {e}")
            return False
    
    # Keep the same interface names for your main.py
    def move_mouse(self, x, y):
        """Use your exact working hardware mouse movement"""
        return self.move_mouse_direct_input(x, y)
    
    def click(self):
        """Use your exact working hardware click"""
        return self.click_hardware()
    
    def get_screen_resolution(self):
        return self.screen_width, self.screen_height