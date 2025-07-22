// dynamic cast
// safe downcasting in class hierarchy
#include <iostream>
using namespace std;

class Animal {
public:
    virtual void sound() {} // virtual function is required to dynamic cast
};

class Dog : public Animal {
public:
    void bark() {
        cout << "Woof!" << endl;
    }
};

int main() {
    Animal* a = new Dog(); // base class pointer to derived class object

    // safe downcast using dynamic cast
    Dog* d = dynamic_cast<Dog*>(a);

    if (d!= nullptr) {
        d->bark(); // call derived class method
    } else {
        cout << "Dynamic cast failed." << endl;
    }

    delete a; // free allocated memory
    return 0;
}
