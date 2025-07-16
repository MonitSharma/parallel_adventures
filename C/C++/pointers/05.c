#include <stdio.h>

int main() {
    int arr[] = {10, 20, 30, 40, 50}; // Declare and initialize an array of integers

    int* ptr = arr; // Declare a pointer and assign it the address of the first element of the array

    printf("Position one: %d\n", *ptr); // Print the value at the first position (10)
    printf("Position two: %d\n", *(ptr + 1)); // Print the value at the second position (20)
    printf("Position three: %d\n", *(ptr + 2)); // Print the value at the third position (30)
    printf("Position four: %d\n", *(ptr + 3)); // Print the value at the fourth position (40)
    printf("Position five: %d\n", *(ptr + 4)); // Print the value at the fifth position (50)

    for (int i =0; i < 5; i ++) {
        printf("%d\t", *ptr); // Print the value at the current position
        printf("Address of position %d: %p\n", i + 1, (void*)ptr); // Print the address of the current position
        ptr++; // Move the pointer to the next position in the array

        // Note: The pointer arithmetic here automatically adjusts the pointer to the next integer in the array
    }

    // Output

        // 10      Address of position 1: 0x7fff627803b0
        // 20      Address of position 2: 0x7fff627803b4    Change of 4
        // 30      Address of position 3: 0x7fff627803b8    Change of 4
        // 40      Address of position 4: 0x7fff627803bc    Change of 4
        // 50      Address of position 5: 0x7fff627803c0    Change of 4

}