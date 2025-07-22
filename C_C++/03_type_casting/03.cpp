// const_cast
// add or remove `const` qualifier safely 
#include <iostream>
using namespace std;

void printValue(int* ptr) {
    *ptr = 100; // valid only if original object is not actually const
    cout << "Value: " << *ptr << endl;

}

int main() {
    int value = 42;
    const int* constPtr = &value; // pointer to const int
    printValue(const_cast<int*>(constPtr)); // remove constness
    cout << "Modified value: " << value << endl;

    return 0;
}