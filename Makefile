CXX = g++
CXXFLAGS = -g -std=c++11 -Wall -Werror=return-type  -Werror=uninitialized # --coverage

SRCS = $(wildcard *.hpp)
OBJECTS = $(SRCS:%.hpp=%.o)

main: $(OBJECTS) main.o
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -rf main.dSYM
	$(RM) *.o *.gc* *.dSYM core main
