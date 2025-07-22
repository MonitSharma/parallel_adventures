#include<stdio.h> // Standard Input Output header file for printf
#include<stdlib.h> // Standard library header file for malloc and free

int main() {
    // NULL pointer this time
    int* ptr = NULL; // Declare a pointer and initialize it to NULL
    printf("Value of ptr: %p\n", (void*)ptr); // Print the value of ptr (should be NULL)

    // Check for the NULL
    if (ptr==NULL) {
        printf("Pointer is NULL, cannot dereference it.\n");

    }

    // Allocate memory for an integer
    ptr = malloc(sizeof(int));
    if (ptr == NULL) {
        printf("Memory allocation failed.\n");
        return 1; // Exit if memory allocation fails
    }

    printf("Memory allocated successfully: %p\n", (void*)ptr); // Print the address of allocated memory

    *ptr = 42; // Assign a value to the allocated memory
    printf("Value pointed to by ptr: %d\n", *ptr); // Print the value pointed to by ptr

    free(ptr); // Free the allocated memory
    ptr = NULL; // Set ptr to NULL after freeing memory to avoid dangling pointer
    printf("Memory freed successfully.\n After freeing, ptr is now: %p\n", (void*)ptr); // Print the value of ptr after freeing memory

    if (ptr == NULL) {
        printf("Pointer is NULL after freeing memory, safe to dereference.\n");
    } else {
        printf("Pointer is not NULL after freeing memory, this is unexpected.\n");
    }

    return 0; // Successful completion
}