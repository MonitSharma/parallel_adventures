#include <stdio.h>
#include <stdlib.h>
int factorial(int n) {
    if (n < 1 ) return 1;

    return n * factorial(n - 1);
}

int main() {
    int x = 5;
    int result = factorial(x);
    printf("Factorial of %d is %d\n", x, result);
    return 0;
}