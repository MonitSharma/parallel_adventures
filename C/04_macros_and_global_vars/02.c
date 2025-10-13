#include <stdio.h>

#ifdef _WIN32
    #define PLATFORM "Windows"
#elif __linux__
    #define PLATFORM "Linux"
#elif __APPLE__
    #define PLATFORM "macOS"
#else
    #define PLATFORM "Unknown"
#endif

int main() {
    printf("Running on %s\n", PLATFORM);
    return 0;
}
