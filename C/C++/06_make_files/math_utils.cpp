#include "math_utils.h"


int square(int x) {
    return x * x;
}

int cube(int x) {
    return x * x * x;
}

int factorial(int x) {
    if (x <= 1) return 1;
    return x * factorial(x - 1);
}

