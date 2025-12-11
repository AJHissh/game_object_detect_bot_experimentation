#ifndef BRIDGE_H
#define BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

// Export functions as C-style to avoid name mangling
__declspec(dllexport) int moveMouse(int x, int y);
__declspec(dllexport) int leftClick(int x, int y);
__declspec(dllexport) int sendInputClick(int x, int y);
__declspec(dllexport) int ctypesClick(int x, int y);
__declspec(dllexport) int gameClick(int x, int y);

#ifdef __cplusplus
}
#endif

#endif