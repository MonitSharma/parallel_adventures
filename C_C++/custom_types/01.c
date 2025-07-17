#include <stdio.h>
#include <stdlib.h>

int main() {
    int arr[] = {12, 34, 56, 78, 90};

    // size_t is used to represent the size of an object
    size_t size = sizeof(arr) / sizeof(arr[0]); // see the readme
    printf("Size of the array: %zu\n", size);                                 
    printf("Size of size_t: %zu bytes\n", sizeof(size_t));
    printf("Size of int: %zu bytes\n", sizeof(int));
    printf("Size of char: %zu bytes\n", sizeof(char));
    printf("Size of double: %zu bytes\n", sizeof(double));
    printf("Size of float: %zu bytes\n", sizeof(float));
    printf("Size of long: %zu bytes\n", sizeof(long));
    printf("Size of long long: %zu bytes\n", sizeof(long long));
    printf("Size of short: %zu bytes\n", sizeof(short));
    printf("Size of long double: %zu bytes\n", sizeof(long double));
    printf("Size of pointer: %zu bytes\n", sizeof(void*));
    printf("Size of function pointer: %zu bytes\n", sizeof(void (*)(void)));
    printf("Size of struct: %zu bytes\n", sizeof(struct { int a; double b; }));
    printf("Size of union: %zu bytes\n", sizeof(union { int a; double b; }));
    printf("Size of enum: %zu bytes\n", sizeof(enum { RED, GREEN, BLUE }));
    printf("Size of typedef: %zu bytes\n", sizeof(size_t));

    return 0;
}

/*
Size of the array: 5
Size of size_t: 8 bytes
Size of int: 4 bytes
Size of char: 1 bytes
Size of double: 8 bytes
Size of float: 4 bytes
Size of long: 8 bytes
Size of long long: 8 bytes
Size of short: 2 bytes
Size of long double: 16 bytes
Size of pointer: 8 bytes
Size of function pointer: 8 bytes
Size of struct: 16 bytes
Size of union: 8 bytes
Size of enum: 4 bytes
Size of typedef: 8 bytes
*/