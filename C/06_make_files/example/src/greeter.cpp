#include "greeter.h"
#include <iostream>
using namespace std;

Greeter::Greeter(const std::string& name) : name(name) {}

void Greeter::say_hello() const {
    cout << "Hello, " << name << "!" << endl;
}