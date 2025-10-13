// reinterpret cast
#include <iostream>
using namespace std;

int main() {
    int num = 65; // ASCII value for 'A'

    // reinterpret int pointer as char pointer
    char* charPtr = reinterpret_cast<char*>(&num);
    cout << "Character representation: " << *charPtr << endl; // prints 'A

    return 0;
}
    
/* only safe for reading, we are not modifying memory

WARNING: reinterpret_cast is dangerous and should be used with caution.
It allows you to treat a pointer of one type as a pointer of another type without any safety
checks. This can lead to undefined behavior if the types are not compatible.
It is primarily used for low-level programming tasks, such as interfacing with hardware or
performing low-level memory manipulation.
It should not be used for type conversions that require safety or correctness checks.
It is not type-safe and can lead to hard-to-debug errors if misused.
Use it only when you are absolutely sure about the types and their memory layout.
It is not recommended for general-purpose type casting in C++.

NEVER USE reinterpret_cast FOR COMPLEX TYPE CONVERSIONS OR MODIFYING MEMORY WITHOUT KNOWING WHAT YOU'RE DOING

*/