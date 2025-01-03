# Compiler and flags
NVCC = nvcc
CFLAGS = -O4 -Xcompiler -Wall -arch=sm_37 -I.
# Targets and dependencies
TARGET = nbody
SRC = nbody.cu
HEADERS = timer.h

# Default target
all: $(TARGET)

# Linking the executable
$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(CFLAGS) -o $@ $(SRC)

# Clean up generated files
clean:
	rm -f $(TARGET)

# Run the program (optional)
run: $(TARGET)
	./$(TARGET)
