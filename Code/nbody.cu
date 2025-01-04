#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
// #include <omp.h>

#define SOFTENING 1e-9f  /* Will guard against denormals */
#define tolerance 0.005// 2 decimals
/* Macro for error checking in CUDA calls */
#define CHECK_CUDA_ERROR(call)                                \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
              __FILE__, __LINE__, cudaGetErrorString(err));   \
      goto cleanup;                                           \
    }                                                         \
  } while (0)
typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {
  for (int i = 0; i < n; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations
  float *buf;
  Body *dbuf;
  Body *hbuf;
  Body *p;
  int bytes = nBodies*sizeof(Body);
  cudaError_t e = cudaSuccess;
  double avgTime = 0.0;
  double totalTime = 0.0;

  buf = (float*)malloc(bytes);
  if(!buf) goto cleanup;
  hbuf = (Body*)malloc(bytes);
  if(!hbuf) goto cleanup;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dbuf, bytes));

  p = (Body*)buf;

  randomizeBodies(buf, 6*nBodies); // Init pos / vel data

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    bodyForce(p, dt, nBodies); // compute interbody forces
    
    // #pragma omp parallel for
    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
  }
  avgTime = totalTime / (double)(nIters-1); 

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
  
  CHECK_CUDA_ERROR(cudaMemcpy(hbuf, dbuf, bytes, cudaMemcpyDeviceToHost));
  
  // Chech similarity between CPU (buf) and GPU (hbuf)
  for (int i = 0; i < nBodies; i++) {
    // Compare positions and velocities between CPU and GPU
    if (fabs(p[i].x - hbuf[i].x) > tolerance ||
        fabs(p[i].y - hbuf[i].y) > tolerance ||
        fabs(p[i].z - hbuf[i].z) > tolerance ||
        fabs(p[i].vx - hbuf[i].vx) > tolerance ||
        fabs(p[i].vy - hbuf[i].vy) > tolerance ||
        fabs(p[i].vz - hbuf[i].vz) > tolerance) {
          printf("Difference bigger than tolerance in index: %d\n", i);
          goto cleanup;
    }
  }
  
  
  
  
  
  cleanup:
  if (buf) free(buf);
  if (dbuf) cudaFree(dbuf);

  cudaDeviceSynchronize();
  e = cudaGetLastError();
  if(e!=cudaSuccess){
    printf("ERROR: %s, FILE: %s, LINE: %d\n", cudaGetErrorString(e), __FILE__, __LINE__);
  }
  else{
    printf("cudaGetLastError() == cudaSuccess!\n");
  }

  cudaDeviceReset();
  
  return 0;
}
