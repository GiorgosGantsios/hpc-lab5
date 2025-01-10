# Compiler settings
CC = icx            # gcc or icx
NVCC = nvcc

# Common flags
CFLAGS = -std=c99 -O3 -fopenmp -D_DEFAULT_SOURCE
NVCC_FLAGS = -O3 -arch=sm_37

# Sources and targets
GPU_SRC = nbody.cu
OMP_SRC = nbody_omp.c
OBJ = nbody_gpu.o
HEADERS = 

# Targets
GPU_TARGET = nbody_gpu
OMP_TARGET = nbody_omp

# Default target
all: $(GPU_TARGET) $(OMP_TARGET)

# GPU target
$(GPU_TARGET): $(GPU_SRC) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $(GPU_SRC) -o nbody_gpu.o
	$(NVCC) $(NVCC_FLAGS) -o $(GPU_TARGET) $(OBJ) -lm

# OpenMP target
$(OMP_TARGET): $(OMP_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $(OMP_TARGET) $(OMP_SRC) -lm

# Clean up
clean:
	rm -f $(GPU_TARGET) $(OMP_TARGET) $(OBJ)

# Run 12 times with optional arguments and save output to file
experiment_12:
	@if [ "$(word 2, $(MAKECMDGOALS))" != "" ]; then \
	  EXEC=$(word 2, $(MAKECMDGOALS)); \
	  ARGS=$(word 3, $(MAKECMDGOALS)); \
	  OUTPUT_FILE=result_$$(echo $$EXEC | cut -d'_' -f2); \
	  echo "Saving output to $$OUTPUT_FILE"; \
	  > $$OUTPUT_FILE; \
	  for i in {1..12}; do \
	    if [ "$$ARGS" != "" ]; then \
	      echo "Running $$i/12: $$EXEC with args $$ARGS" | tee -a $$OUTPUT_FILE; \
	      ./$$EXEC $$ARGS >> $$OUTPUT_FILE; \
	    else \
	      echo "Running $$i/12: $$EXEC with default args" | tee -a $$OUTPUT_FILE; \
	      ./$$EXEC >> $$OUTPUT_FILE; \
	    fi; \
	  done; \
	else \
	  echo "Error: Specify the executable and optionally the arguments, e.g., 'make experiment_12 nbody_omp 128000' or 'make experiment_12 nbody_omp'"; \
	fi

# Prevent Makefile from treating extra arguments as targets
%:
	@:
