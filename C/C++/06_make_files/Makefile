CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

OBJS = main.o math_utils.o


app: $(OBJS)
	$(CXX) $(CXXFLAGS) -o app $(OBJS)


main.o: main.cpp math_utils.h
	$(CXX) $(CXXFLAGS) -c main.cpp

math_utils.o: math_utils.cpp math_utils.h
	$(CXX) $(CXXFLAGS) -c math_utils.cpp

clean:
	rm -f *.o app
.PHONY: clean