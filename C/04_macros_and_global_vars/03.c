#include <stdio.h>

#ifndef BUFFER_SIZE
    #define BUFFER_SIZE 1024 // Default buffer size
#endif

char buffer[BUFFER_SIZE];

int main() {
    printf("Buffer size is set to %d bytes.\n", BUFFER_SIZE);
    
    // Example usage of the buffer
    snprintf(buffer, BUFFER_SIZE, "This is a buffer of size %d bytes.", BUFFER_SIZE);
    printf("%s\n", buffer);

    return 0;
}