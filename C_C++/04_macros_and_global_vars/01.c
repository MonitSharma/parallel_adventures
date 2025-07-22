// enable or disable debug print
#include <stdio.h>

// Toggle this to enable or disable debug prints
// 0 = disable, 1 = enable
#define DEBUG 1

#if DEBUG
#define DEBUG_PRINT(x) printf("DEBUG: %s\n", x)

#else
#define DEBUG_PRINT(x) // No operation
#endif

int main() {
    DEBUG_PRINT("This is a debug message.");

    // Your main code logic here
    printf("Hello, World!\n");

    return 0;
}