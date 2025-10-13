#include <stdio.h> // Standard Input Output header file for printf

int main() {
    int x = 10; // Declare an integer variable x and initialize it to 10
    int* p = &x; // Declare a pointer p and assign it the address of, & says "address of" x
    printf("Value of x: %d\n", x); // Print the value of x
    printf("Address of x: %p\n", (void*)&x); // Print the address of x
    printf("Value of p (address of x): %p\n", (void*)p); // Print the value of p (address of x)
    printf("Value pointed to by p: %d\n", *p); // Print the value pointed to by p (value of x), this is a dereference operation
    return 0;
}