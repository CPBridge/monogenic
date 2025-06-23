# Compiler (g++)
CPP:=g++

# Include directory
INCLUDE_DIR:=-I../../include/

# Source path
VPATH:=../../src/

# Compiler Flags (warnings, C++11, OpenMP, optimisation)
CPPFLAGS:=-c -Wall -Wextra -std=c++11 -fopenmp -O2 $(INCLUDE_DIR)

# Linker Flags (OpenMP, OpenCV, Boost Program Options)
LDFLAGS1:=-fopenmp
LDFLAGS2:=`pkg-config --libs opencv4`

# Name of the executable
EXEC:=monogenicTest

# Top level target
$(EXEC): monogenicTest.o monogenicProcessor.o
	$(CPP) $(LDFLAGS1) $^ -o $@ $(LDFLAGS2)

# Object files
%.o: %.cpp
	$(CPP) $(CPPFLAGS) $< -o $@

# Clean
clean:
	rm $(EXEC) *.o
