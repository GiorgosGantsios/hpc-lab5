#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#define TILE_SIZE 1024
// #include <omp.h>

#define SOFTENING 1e-9f  /* Will guard against denormals */
#define tolerance 1// 2 decimals
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

// __global__ void bodyForce_gpu(Body *p, float dt, int n) {
//   float distSqr;
//   float invDist;
//   float invDist3;
//   int j;//, i;
//   float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

//   int index = blockIdx.x*blockDim.x + threadIdx.x;
//   if(index < n){
//     // Calculate constituted force
//     for (j = 0; j < n; j++) {
//         float dx = p[j].x - p[index].x;
//         float dy = p[j].y - p[index].y;
//         float dz = p[j].z - p[index].z;
//         distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
//         invDist = 1.0f / sqrtf(distSqr);
//         invDist3 = invDist * invDist * invDist;

//         Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
//     }

//     // Update velocity
//     p[index].vx += dt*Fx; p[index].vy += dt*Fy; p[index].vz += dt*Fz;

//     __syncthreads();
//     // Update position
//     p[index].x += p[index].vx*dt;
//     p[index].y += p[index].vy*dt;
//     p[index].z += p[index].vz*dt;
//   }
// }

// bodyForce_gpu_tiled(dbuf_hor, dbuf_ver, dFx+offset, dFy+offset, dFz+offset, dt, TILE_SIZE)
__global__ void bodyForce_gpu_tiled(Body *p_hor, Body *p_ver, float *Fx, float *Fy, float *Fz, float dt, int n) {

  float distSqr;
  float invDist;
  float invDist3;
  int j;//, i;

  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index < n){
    // Calculate constituted force
    for (j = 0; j < n; j++) {
        float dx = p_hor[j].x - p_ver[index].x;
        float dy = p_hor[j].y - p_ver[index].y;
        float dz = p_hor[j].z - p_ver[index].z;
        distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        invDist = 1.0f / sqrtf(distSqr);
        invDist3 = invDist * invDist * invDist;

        Fx[index] += dx * invDist3; Fy[index] += dy * invDist3; Fz[index] += dz * invDist3;
    }
  }
}

__global__ void calculate_pos_vel(Body *p_ver, float *Fx, float *Fy, float *Fz, float dt, int n){
  // Update velocity
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < n){
      p_ver[index].vx += dt*Fx[index]; p_ver[index].vy += dt*Fy[index]; p_ver[index].vz += dt*Fz[index];

      // __syncthreads();
      // Update position
      p_ver[index].x += p_ver[index].vx*dt;
      p_ver[index].y += p_ver[index].vy*dt;
      p_ver[index].z += p_ver[index].vz*dt;
    }
}


int check_correctness(Body *hbuf, Body *p, int nBodies){
  for (int i = 0; i < nBodies; i++) {
    // Compare positions and velocities between CPU and GPU
    if (fabs(p[i].x - hbuf[i].x) > tolerance ||
        fabs(p[i].y - hbuf[i].y) > tolerance ||
        fabs(p[i].z - hbuf[i].z) > tolerance ||
        fabs(p[i].vx - hbuf[i].vx) > tolerance ||
        fabs(p[i].vy - hbuf[i].vy) > tolerance ||
        fabs(p[i].vz - hbuf[i].vz) > tolerance) {
        
        printf("Difference exceeds tolerance at index: %d\n", i);
        printf("Position - CPU: (%.6f, %.6f, %.6f), GPU: (%.6f, %.6f, %.6f)\n",
               p[i].x, p[i].y, p[i].z, hbuf[i].x, hbuf[i].y, hbuf[i].z);
        printf("Velocity - CPU: (%.6f, %.6f, %.6f), GPU: (%.6f, %.6f, %.6f)\n",
               p[i].vx, p[i].vy, p[i].vz, hbuf[i].vx, hbuf[i].vy, hbuf[i].vz);
        return 0;
    }
  }
  return 1;
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations
  float *buf;
  Body *dbuf;
  Body *dbuf_hor, *dbuf_ver;
  float *dFx, *dFy, *dFz;
  Body *hbuf;
  Body *p;
  int bytes = nBodies*sizeof(Body);
  cudaError_t e = cudaSuccess;
  cudaEvent_t startCuda, stopCuda;
  double avgTime = 0.0;
  double totalTime = 0.0;
  int extra_block;
  float msec = 0.0;

  buf = (float*)malloc(bytes);
  if(!buf) goto cleanup;
  hbuf = (Body*)malloc(bytes);
  if(!hbuf) goto cleanup;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dbuf, bytes));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dbuf_hor, TILE_SIZE*sizeof(Body)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dbuf_ver, TILE_SIZE*sizeof(Body)));
  
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dFx, nBodies*nBodies*sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dFy, nBodies*nBodies*sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dFz, nBodies*nBodies*sizeof(float)));
  

  p = (Body*)buf;

  randomizeBodies(buf, 6*nBodies); // Init pos / vel data

  CHECK_CUDA_ERROR(cudaMemcpy(dbuf, p, bytes, cudaMemcpyHostToDevice));

  // CPU Implementation
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
  
  
  extra_block = (nBodies%1024 != 0);
  
  //GPU Impkementation
  cudaEventCreate(&startCuda);
  cudaEventCreate(&stopCuda);

  cudaEventRecord(startCuda, 0);
  for (int iter = 1; iter <= nIters; iter++) {
    // bodyForce_gpu<<<(nBodies/1024)+extra_block, 1024>>>(dbuf, dt, nBodies);
    CHECK_CUDA_ERROR(cudaMemset(dFx, 0, nBodies*nBodies*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(dFy, 0, nBodies*nBodies*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemset(dFz, 0, nBodies*nBodies*sizeof(float)));
    for(int i = 0; i < nBodies; i += TILE_SIZE){
      int j = 0;
      int offset = i*nBodies+j;
      CHECK_CUDA_ERROR(cudaMemcpy(dbuf_ver, dbuf + i, (TILE_SIZE*sizeof(Body) < bytes) ? TILE_SIZE*sizeof(Body): bytes, cudaMemcpyDeviceToDevice));
      
      for(j = 0; j < nBodies; j += TILE_SIZE){
        CHECK_CUDA_ERROR(cudaMemcpy(dbuf_hor, dbuf + j, (TILE_SIZE*sizeof(Body) < bytes) ? TILE_SIZE*sizeof(Body): bytes, cudaMemcpyDeviceToDevice));
        offset = i*nBodies+j;
        bodyForce_gpu_tiled<<<(nBodies/TILE_SIZE) + (nBodies % TILE_SIZE > 0 ? 1 : 0), TILE_SIZE>>>(dbuf_hor, dbuf_ver, dFx+offset, dFy+offset, dFz+offset, dt, TILE_SIZE);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if(err!=cudaSuccess){
          printf("ERROR: %s, FILE: %s, LINE: %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
          goto cleanup;
        }
        else{
          printf("cudaGetLastError() == cudaSuccess! iter = %d, i = %d, j = %d\n", iter, i, j);
        }
      }
      calculate_pos_vel<<<(nBodies/TILE_SIZE) + (nBodies % TILE_SIZE > 0 ? 1 : 0), TILE_SIZE>>>(dbuf_ver, dFx+offset, dFy+offset, dFz+offset, dt, TILE_SIZE);
      cudaDeviceSynchronize();
      cudaError_t err = cudaGetLastError();
      if(err!=cudaSuccess){
        printf("ERROR: %s, FILE: %s, LINE: %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        goto cleanup;
      }
      else{
        printf("cudaGetLastError() == cudaSuccess!\n");
      }
      CHECK_CUDA_ERROR(cudaMemcpy(dbuf+i, dbuf_ver, (TILE_SIZE*sizeof(Body) < bytes) ? TILE_SIZE*sizeof(Body): bytes , cudaMemcpyDeviceToDevice));
      CHECK_CUDA_ERROR(cudaMemcpy(hbuf+i, dbuf_ver, (TILE_SIZE*sizeof(Body) < bytes) ? TILE_SIZE*sizeof(Body): bytes , cudaMemcpyDeviceToHost));
    }
  }

  // CHECK_CUDA_ERROR(cudaMemcpy(hbuf, dbuf, bytes, cudaMemcpyDeviceToHost));
  cudaEventRecord(stopCuda, 0);
  cudaEventSynchronize(stopCuda);
  cudaEventElapsedTime(&msec, startCuda, stopCuda);
  printf("GPU TIME: %f\n", msec);
  // Chech similarity between CPU (buf) and GPU (hbuf)
  if(check_correctness(hbuf, p, nBodies)){
    printf("SUCCESS: Results are the same!\n");
  }
  else{
    printf("FAIL: Results are diffesrent\n");
    goto cleanup;
  }
  
  cleanup:
  if (buf) free(buf);
  if (hbuf) free(hbuf);
  if (dbuf) cudaFree(dbuf);
  if (dbuf_hor) cudaFree(dbuf_hor);
  if (dbuf_ver) cudaFree(dbuf_ver);
  if (dFx) cudaFree(dFx);
  if (dFy) cudaFree(dFy);
  if (dFz) cudaFree(dFz);

  CHECK_CUDA_ERROR(cudaEventDestroy(startCuda));
  CHECK_CUDA_ERROR(cudaEventDestroy(stopCuda));

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
