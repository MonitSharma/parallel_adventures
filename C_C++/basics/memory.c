#include <stdio.h>

int main() {
    int myInt;
    float myFloat;
    char myChar;
    double myDouble;


    printf("Size of int: %zu bytes\n", sizeof(myInt));
    printf("Size of float: %zu bytes\n", sizeof(myFloat));
    printf("Size of char: %zu bytes\n", sizeof(myChar));
    printf("Size of double: %zu bytes\n", sizeof(myDouble));
    printf("Size of int pointer: %zu bytes\n", sizeof(int*));
    printf("Size of float pointer: %zu bytes\n", sizeof(float*));
    printf("Size of char pointer: %zu bytes\n", sizeof(char*));
    printf("Size of double pointer: %zu bytes\n", sizeof(double*));
    printf("Size of void pointer: %zu bytes\n", sizeof(void*));

    return 0;
}

/*
Size of int: 4 bytes
Size of float: 4 bytes
Size of char: 1 bytes
Size of double: 8 bytes
Size of int pointer: 8 bytes
Size of float pointer: 8 bytes
Size of char pointer: 8 bytes
Size of double pointer: 8 bytes
Size of void pointer: 8 bytes
*/