# advanced_click_controller.py
import pyautogui
import time
import ctypes
import win32api
import win32con
import random
from enum import Enum

class ClickMethod(Enum):
    PYTHON_AUTOGUI = "pyautogui"
    PYTHON_AUTOGUI_SLOW = "pyautogui_slow"
    CTYPES_FAST = "ctypes_fast"
    CTYPES_SLOW = "ctypes_slow"
    CTYPES_GAME = "ctypes_game"
    WIN32API = "win32api"
    SENDINPUT = "sendinput"
    DIRECTX = "directx"
    RAPID = "rapid"
    PRESSURE = "pressure"

class AdvancedClickController:
    def __init__(self, default_method=ClickMethod.CTYPES_GAME):
        self.screen_width, self.screen_height = pyautogui.size()
        self.default_method = default_method
        self.setup_methods()
        
    def setup_methods(self):
        self.methods = {
            ClickMethod.PYTHON_AUTOGUI: self._click_pyautogui,
            ClickMethod.PYTHON_AUTOGUI_SLOW: self._click_pyautogui_slow,
            ClickMethod.CTYPES_FAST: self._click_ctypes_fast,
            ClickMethod.CTYPES_SLOW: self._click_ctypes_slow,
            ClickMethod.CTYPES_GAME: self._click_ctypes_game,
            ClickMethod.WIN32API: self._click_win32api,
            ClickMethod.SENDINPUT: self._click_sendinput,
            ClickMethod.DIRECTX: self._click_directx,
            ClickMethod.RAPID: self._click_rapid,
            ClickMethod.PRESSURE: self._click_pressure,
        }
    
    def move_mouse(self, x, y, method='ctypes'):
        """Move mouse using specified method"""
        try:
            x = max(0, min(self.screen_width - 1, int(x)))
            y = max(0, min(self.screen_height - 1, int(y)))
            
            if method == 'pyautogui':
                pyautogui.moveTo(x, y, duration=0.1)
            else:  # ctypes (default)
                ctypes.windll.user32.SetCursorPos(x, y)
            
            time.sleep(0.05)  
            return True
        except Exception as e:
            print(f"Mouse move error: {e}")
            return False
    
    def _click_pyautogui(self, x, y):
        pyautogui.click()
        return True
    
    def _click_pyautogui_slow(self, x, y):
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.mouseUp()
        return True
    
    def _click_ctypes_fast(self, x, y):
        ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)
        return True
    
    def _click_ctypes_slow(self, x, y):
        ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
        time.sleep(0.1)
        ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)
        return True
    
    def _click_ctypes_game(self, x, y):
        ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
        time.sleep(0.15)
        ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)
        return True
    
    def _click_win32api(self, x, y):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
        return True
    
    def _click_sendinput(self, x, y):
        class INPUT(ctypes.Structure):
            class _INPUT(ctypes.Union):
                _fields_ = [("mi", ctypes.c_ulong * 6)]
            _anonymous_ = ("_input",)
            _fields_ = [("type", ctypes.c_ulong), ("_input", _INPUT)]
        
        mouse_down = INPUT()
        mouse_down.type = 0
        mouse_down.mi[3] = 0x0002
        
        mouse_up = INPUT()
        mouse_up.type = 0
        mouse_up.mi[3] = 0x0004
        
        ctypes.windll.user32.SendInput(1, ctypes.byref(mouse_down), ctypes.sizeof(INPUT))
        time.sleep(0.1)
        ctypes.windll.user32.SendInput(1, ctypes.byref(mouse_up), ctypes.sizeof(INPUT))
        return True
    
    def _click_directx(self, x, y):
        ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
        time.sleep(0.2)
        ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)
        return True
    
    def _click_rapid(self, x, y):
        for i in range(2):  # Double click rapidly
            ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
            ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)
            time.sleep(0.02)
        return True
    
    def _click_pressure(self, x, y):
        ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
        time.sleep(0.3)  # Long press
        ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)
        return True
    
    def click(self, x=None, y=None, method=None):
        """Perform click using specified method"""
        if method is None:
            method = self.default_method
        
        try:
            if x is not None and y is not None:
                if not self.move_mouse(x, y):
                    return False
            
            click_func = self.methods.get(method, self._click_ctypes_game)
            return click_func(x or 0, y or 0)
            
        except Exception as e:
            print(f"Click error ({method}): {e}")
            return False
    
    def test_all_methods(self):
        print("Testing all click methods...")
        test_x = self.screen_width // 2 + 100
        test_y = self.screen_height // 2
        
        results = {}
        for method in ClickMethod:
            print(f"Testing {method.value}...")
            success = self.click(test_x, test_y, method)
            results[method.value] = success
            time.sleep(0.5)
        
        print("\nResults:")
        for method, success in results.items():
            status = "✅ WORKING" if success else "❌ FAILED"
            print(f"{method}: {status}")
        
        return results

if __name__ == "__main__":
    controller = AdvancedClickController()
    
    print("Advanced Click Controller Test")
    print("=" * 40)
    
    results = controller.test_all_methods()
    
    working = [method for method, success in results.items() if success]
    print(f"\nWorking methods: {working}")