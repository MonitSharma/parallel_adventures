#include <stdio.h>


int main() {
    int arr1[] = {10, 20, 30, 40, 50}; // Declare and initialize an array of integers
    int arr2[] = {60, 70, 80, 90, 100}; // Another array of integers

    int* ptr1 = arr1;
    int* ptr2 = arr2; // Declare pointers and assign them the addresses of the first elements of the arrays

    int* matrix[] = {ptr1, ptr2}; // Create an array of pointers to the first elements of the two arrays

    for (int i =0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
            printf("Value at matrix[%d][%d]: %d\n", i, j, *(matrix[i] + j)); // Print the value at the current position in the matrix
            printf("Address of matrix[%d][%d]: %p\n", i, j, (void*)(matrix[i] + j)); // Print the address of the current position in the matrix
        }
        printf("\n"); // Print a newline for better readability
    }
}