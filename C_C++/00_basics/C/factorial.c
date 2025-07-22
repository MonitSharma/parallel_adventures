// Write what this program does

// Program : Calculate the factorial of a number

// Preprocessor Directives : Gives instructions to the compiler
#include <stdio.h>  // Include standard input-output header file

// Section : Function Declaration, defining some constants and macros that can be used later
#define MAX 12  // Define a constant MAX with value 100

// Global Decalaration : These can be accessed and modified by any function in the program
int global_var; 

// Function to calculate factorial of a number
int factorial(int n) {
    int fact = 1;  // Initialize factorial variable
    for (int i =1; i <= n; i++) { 
        fact *= i;  // Multiply fact by i for each iteration
    }
    return fact;  // Return the calculated factorial    
}


// main function : Entry point of the program
int main() {
    int n;
    printf("Enter a number to calculate its factorial: ");
    scanf("%d", &n);  // Read an integer from user input
    if (n < 0 || n > MAX) {  // Check if the input is valid
        printf("Please enter a number between 0 and %d\n", MAX);
        return 1;  // Return 1 to indicate an error
    }
    printf("Factorial of %d is %d\n", n, factorial(n));  // Print the factorial of the number
    return 0;  // Return 0 to indicate successful execution
}


