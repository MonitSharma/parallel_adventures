// macro for logging

#include <stdio.h>

#define LOG(msg) printf("[LOG] %s:%d - %s\n", __FILE__, __LINE__, msg)

int main() {
    LOG("Application started");
    // Some code...
    LOG("Application ended");
    return 0;
}
