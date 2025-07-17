#include <stdio.h>

typedef struct {
    float x;
    float y;
} Point;

int main () {
    Point p = {3.5, 4.5};
    printf("Point coordinates: (%.2f, %.2f)\n", p.x, p.y);
    printf("Size of Point struct: %zu bytes\n", sizeof(Point));
}
