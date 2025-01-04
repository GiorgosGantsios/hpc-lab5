#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f  /* Will guard against denormals */

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

// kapakos failure code
// void randomizeBodies(float *x, float *y, float *z, float *vx, float *vy, float *vz, int n) {

//   for (int i=0; i<n; i++) {
//     x[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//     y[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//     z[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

//     vx[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//     vy[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//     vz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//   }
// }

// void bodyForce(Body *p, float dt, int n) {
//   float distSqr;
//   float invDist;
//   float invDist3;
//   int j, i;
//   float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
//   //float vx = 0.0, vy = 0.0, vz = 0.0;

//   #pragma omp parallel for schedule(static) private(distSqr, invDist, invDist3, j) reduction(+: Fx) reduction(+: Fy) reduction(+: Fz) //reduction(+: vx) reduction(+: vy) reduction(+: vz) reduction(+: p[:n]) reduction(+: p[:n])
//   for (i = 0; i < n; i++) { 
//     Fx = 0.0f;
//     Fy = 0.0f;
//     Fz = 0.0f;
//     for (j = 0; j < n; j++) {
//         float dx = p[j].x - p[i].x;
//         float dy = p[j].y - p[i].y;
//         float dz = p[j].z - p[i].z;
//         distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
//         invDist = 1.0f / sqrtf(distSqr);
//         invDist3 = invDist * invDist * invDist;

//         Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
//     }
//     // vx = p[i].vx + dt*Fx;
//     // vy = p[i].vy + dt*Fy;
//     // vz = p[i].vz + dt*Fz;
//     // #pragma omp critical
//     // {
//     //   p[i].vx = vx;
//     //   p[i].vy = vy;
//     //   p[i].vz = vz;
//     // }

//     p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
//   }
// }

__global__ void bodyForce(Body *p, float dt, int n) {
  float distSqr;
  float invDist;
  float invDist3;
  int j, i;
  float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

  for (i = 0; i < n; i++) { 
    Fx = 0.0f;
    Fy = 0.0f;
    Fz = 0.0f;
    for (j = 0; j < n; j++) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        invDist = 1.0f / sqrtf(distSqr);
        invDist3 = invDist * invDist * invDist;

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

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;
  // p = (Body*)malloc(bytes);

  float *d_buf;
  cudaMalloc(&d_buf,bytes);
  Body *dp = (Body*)d_buf;
  

  // cudaError_t err = cudaMalloc((void**)&dp, sizeof(Body));
  // if (err != cudaSuccess) {
  //     fprintf(stderr, "CUDA malloc error: %s\n", cudaGetErrorString(err));
  //     return(-1);
  // }
  //cudaMemcpy(dp, p, bytes, cudaMemcpyHostToDevice);

  randomizeBodies(buf, 6*nBodies); // Init pos / vel data

  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();
    cudaMemcpy(dp, p, bytes, cudaMemcpyHostToDevice);
    bodyForce<<<(nBodies/1024)+1, 1024>>>(dp, dt, nBodies); // compute interbody forces
    cudaMemcpy(p, dp, bytes, cudaMemcpyDeviceToHost);
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
  double avgTime = totalTime / (double)(nIters-1); 

  for (int i = 0; i < nBodies; i++)  {
    printf("x: %f, y: %f, z: %f\n", p[i].x, p[i].y, p[i].z);
  }

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
  // free(buf);
}