#include <stdio.h> // Standard Input Output header file for printf


int main() {
    int value = 20; // Declare an integer variable value and initialize it to 20
    int* ptr1 = &value; // Declare a pointer ptr and assign it the address of value   ptr1 to value
    int** ptr2 = &ptr1; // Declare a pointer to pointer ptr2 and assign it the address of ptr1 , ptr2 to ptr1 to value
    int*** ptr3 = &ptr2; // Declare a pointer to pointer to pointer ptr3 and assign it the address of ptr2
    
    
    printf("Value of value: %d\n", value); // Print the value of value
    printf("Address of value: %p\n", (void*)&value); // Print the
    printf("Value of ptr1 (address of value): %p\n", (void*)ptr1); // Print the value of ptr1 (address of value)
    printf("Value pointed to by ptr1: %d\n", *ptr1); // Print the value pointed to by ptr1 (value of value)
    printf("Value of ptr2 (address of ptr1): %p\n", (void*)ptr2); // Print the value of ptr2 (address of ptr1)
    printf("Value pointed to by ptr2 (address of value): %p\n", (void*)*ptr2); // Print the value pointed to by ptr2 (address of value)
    printf("Value pointed to by ptr2: %d\n", **ptr2); // Print the value pointed to by ptr2 (value of value)
    printf("Value of ptr3 (address of ptr2): %p\n", (void*)ptr3); // Print the value of ptr3 (address of ptr2)
    printf("Value pointed to by ptr3 (address of ptr1): %p\n", (void*)*ptr3); // Print the value pointed to by ptr3 (address of ptr1)
    printf("Value pointed to by ptr3 (address of value): %p\n", (void*)**ptr3); // Print the value pointed to by ptr3 (address of value)
    printf("Value pointed to by ptr3: %d\n", ***ptr3); // Print the value pointed to by ptr3 (value of value)
    return 0; // Return 0 to indicate successful execution
}