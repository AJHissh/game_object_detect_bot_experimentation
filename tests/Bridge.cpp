#include "Bridge.h"
#include <windows.h>
#include <thread>
#include <chrono>

// Use extern "C" to prevent C++ name mangling
extern "C" {
    __declspec(dllexport) int moveMouse(int x, int y) {
        return SetCursorPos(x, y) ? 1 : 0;
    }

    __declspec(dllexport) int leftClick(int x, int y) {
        if (!SetCursorPos(x, y)) return 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        INPUT inputDown = {0};
        inputDown.type = INPUT_MOUSE;
        inputDown.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        
        INPUT inputUp = {0};
        inputUp.type = INPUT_MOUSE;
        inputUp.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        
        SendInput(1, &inputDown, sizeof(INPUT));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        SendInput(1, &inputUp, sizeof(INPUT));
        
        return 1;
    }

    __declspec(dllexport) int sendInputClick(int x, int y) {
        if (!SetCursorPos(x, y)) return 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        INPUT inputs[2] = {0};
        
        inputs[0].type = INPUT_MOUSE;
        inputs[0].mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        
        inputs[1].type = INPUT_MOUSE;
        inputs[1].mi.dwFlags = MOUSEEVENTF_LEFTUP;
        
        UINT result = SendInput(2, inputs, sizeof(INPUT));
        return result == 2 ? 1 : 0;
    }

    __declspec(dllexport) int ctypesClick(int x, int y) {
        if (!SetCursorPos(x, y)) return 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
        
        return 1;
    }

    __declspec(dllexport) int gameClick(int x, int y) {
        if (!SetCursorPos(x, y)) return 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        
        INPUT inputs[2] = {0};
        
        inputs[0].type = INPUT_MOUSE;
        inputs[0].mi.dwFlags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTDOWN;
        inputs[0].mi.dx = x * 65536 / GetSystemMetrics(SM_CXSCREEN);
        inputs[0].mi.dy = y * 65536 / GetSystemMetrics(SM_CYSCREEN);
        
        inputs[1].type = INPUT_MOUSE;
        inputs[1].mi.dwFlags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTUP;
        inputs[1].mi.dx = x * 65536 / GetSystemMetrics(SM_CXSCREEN);
        inputs[1].mi.dy = y * 65536 / GetSystemMetrics(SM_CYSCREEN);
        
        UINT result = SendInput(2, inputs, sizeof(INPUT));
        return result == 2 ? 1 : 0;
    }
}