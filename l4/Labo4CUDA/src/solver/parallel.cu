#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "parallel.cuh"

#define ELEMENTS 5

#define errCheck(code) { errorCheck((code), __FILE__, __LINE__); }
void addWithCuda(const int* a, const int* b, int* c, int elements);

using std::cout;
using std::flush;
using std::endl;

inline void errorCheck(cudaError_t code, const char* file, int line) {
    if (cudaSuccess != code) {
        std::cout << "[" << file << ", line " << line << "]" << std::flush;
        std::cout << " CUDA error <" << cudaGetErrorString(code) << "> received." << std::endl << std::flush;
        exit(EXIT_FAILURE);
    }
}

__global__ void addKernel(const int* a, const int* b, int* c, int elements) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    printf("x = %d, BlockIdx = %d, BlockDimx = %d, threadIdx = %d\n", x, blockIdx.x, blockDim.x, threadIdx.x);
    if (x < elements) {
        c[x] = a[x] + b[x];
    }
}

void solvePar(int rows, int cols, int iterations, double td, double h, double** matrix) {
    cout << "Do cuda related stuff here!" << endl << flush;

    // Example.
    // int elements = 5;
    const int a[ELEMENTS] = { 1, 2, 3, 4, 5 };
    const int b[ELEMENTS] = { 5, 4, 3, 2, 1 };
    int c[ELEMENTS] = { 0 };

    addWithCuda(a, b, c, ELEMENTS);
}


// before in lab 3 (too complex in the end)
    // get initial totalMatrix
    // extract partialMatrix
    // in the main calculation method
    //      for each iteration
    //          ||calculate the heatmap of partialMatrix||
    //      end for loop
    // copy/paste the values back to totalMatrix

// AFTER in lab 3 (this final method)
    // in the main calculation method
    //      for each iteration
    //          get initial totalMatrix
    //          extract partialMatrix
    //              ||calculate the heatmap of partialMatrix||
    //              sync
    //          copy/paste the values back to totalMatrix
    //      end for loop

// for lab 4
// for each iteration
//      get totalMatrix
//      extract partialMatrix
//              || parallel calculate partialMatrix ||
//      copy paste partial to total
// end for
void addWithCuda(const int* a, const int* b, int* c, int elements) {
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(5, 1, 1);

    errCheck(cudaSetDevice(0));

    errCheck(cudaMalloc((void**)&dev_c, elements * sizeof(int)));
    errCheck(cudaMalloc((void**)&dev_a, elements * sizeof(int)));
    errCheck(cudaMalloc((void**)&dev_b, elements * sizeof(int)));

    errCheck(cudaMemcpy(dev_a, a, elements * sizeof(int), cudaMemcpyHostToDevice));
    errCheck(cudaMemcpy(dev_b, b, elements * sizeof(int), cudaMemcpyHostToDevice));
    
    // for should be here
    // extract partial matrix from total matrix

    // in our addKernel, arguments should be
    // totalInputMatrix, partialMatrix, totalOutputMatrix
    addKernel << <dimGrid, dimBlock >> > (dev_a, dev_b, dev_c, elements);

    errCheck(cudaGetLastError());
    errCheck(cudaDeviceSynchronize());
    errCheck(cudaMemcpy(c, dev_c, elements * sizeof(int), cudaMemcpyDeviceToHost));
    errCheck(cudaFree(dev_a));
    errCheck(cudaFree(dev_b));
    errCheck(cudaFree(dev_c));
    errCheck(cudaDeviceReset());

    cout << "c = { " << c[0] << flush;
    for (int i = 1; i < elements; i++) {
        cout << ", " << c[i] << flush;
    }

    cout << " }" << endl << flush;
}
