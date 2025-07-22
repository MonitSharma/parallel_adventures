// static_cast
#include <iostream>
using namespace std;

int main() {
    double pi = 3.14159;

    // safely converting double to int
    int intPi = static_cast<int>(pi);
    cout << "Original double value: " << pi << endl;
    cout << "Converted int value: " << intPi << endl;

    return 0;
}