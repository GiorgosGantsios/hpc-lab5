# Compiler settings
CC = gcc            # Change to icx if needed
NVCC = nvcc

# Common flags
CFLAGS = -std=c99 -O3 -fopenmp -D_DEFAULT_SOURCE
NVCC_FLAGS = -arch=sm_37 -O3

# Sources and targets
CPU_SRC = nbody.c
GPU_SRC = nbody.cu
OBJ = nbody.o nbody_gpu.o
HEADERS = 

# Targets
CPU_TARGET = nbody_cpu
GPU_TARGET = nbody_gpu

# Default target
all: $(CPU_TARGET) $(GPU_TARGET)

# CPU target
$(CPU_TARGET): $(CPU_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -c $(CPU_SRC) -o nbody.o
	$(CC) $(CFLAGS) -o $(CPU_TARGET) nbody.o -lm

# GPU target
$(GPU_TARGET): $(CPU_SRC) $(GPU_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -c $(CPU_SRC) -o nbody.o
	$(NVCC) $(NVCC_FLAGS) -c $(GPU_SRC) -o nbody_gpu.o
	$(NVCC) $(NVCC_FLAGS) -o $(GPU_TARGET) $(OBJ) -lm

# Clean up
clean:
	rm -f $(CPU_TARGET) $(GPU_TARGET) $(OBJ)
