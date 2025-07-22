#include <stdio.h> // Standard Input Output header file for printf

int main() {
    int num = 10;
    float fnum = 3.14;
    void* vptr; // Declare a void pointer

    vptr = &num; // Assign the address of num to the void pointer
    printf("Integer value: %d\n", *(int*)vptr); // Cast void pointer to int pointer and dereference it
    // why does vptr prints the value of num?
    // Because we cast the void pointer to an int pointer before dereferencing it, allowing us
    // to access the integer value stored at that address.


    vptr = &fnum; // Assign the address of fnum to the void pointer
    printf("Float value: %.2f\n", *(float*)vptr); // Cast void pointer to float pointer and dereference it
    // why does vptr prints the value of fnum?
    // Because we cast the void pointer to a float pointer before dereferencing it,
    // allowing us to access the float value stored at that address.
}