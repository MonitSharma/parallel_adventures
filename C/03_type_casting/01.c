#include <stdio.h>  

int main() {
    float f = 69.69;

    int i = (int)f; 

    printf("Original float value: %.2f\n", f);
    printf("Converted int value: %d\n", i);

    // now to char
    char c = (char)i; // converting int to char
    printf("Converted char value: %c\n", c); // prints 'E' (ASCII value 69)

    
}