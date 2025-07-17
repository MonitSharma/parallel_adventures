#include <iostream>

using namespace std;

typedef struct {
    float x;
    float y;
} Point;

int main() {
    Point p = {3.5, 4.5};
    cout << "Point coordinates: (" << p.x << ", " << p.y << ")" << endl;
    cout << "Size of Point struct: " << sizeof(Point) << " bytes" << endl;

    return 0;
}