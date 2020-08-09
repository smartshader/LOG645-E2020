#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "parallel.cuh"

//#define ELEMENTS 5

#define errCheck(code) { errorCheck((code), __FILE__, __LINE__); }
void addWithCuda(int rows, int cols, int iterations, double td, double h, double** matrix);
void checkLocalDevice();

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

__global__ void addKernel(double* a, double* c, int elements) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    printf("x = %d, BlockIdx = %d, BlockDimx = %d, threadIdx = %d\n", x, blockIdx.x, blockDim.x, threadIdx.x);
    if (x < elements) {
        //c[x] = a[x] + b[x];
        c[x] = a[x];
    }
}

void solvePar(int rows, int cols, int iterations, double td, double h, double** matrix) {
    // Example.
    // int elements = 5;
    //double a[ELEMENTS] = { 1, 2, 3, 4, 5 };
    //double b[ELEMENTS] = { 5, 4, 3, 2, 1 };
    //double c[ELEMENTS] = { 0 };

    addWithCuda(rows, cols, iterations, td, h, matrix);

    checkLocalDevice();
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
void addWithCuda(int rows, int cols, int iterations, double td, double h, double** matrix) {
    double* dev_matrix = nullptr;
    double* dev_subMatrix = nullptr;



    // calculate partial rows and cols matrix
    int partialRow = (rows % 2 == 0) ? rows / 2 : rows / 2 + 1;
    int partialCol = (cols % 2 == 0) ? cols / 2 : cols / 2 + 1;

    // calculate the total number of tiles to process, instanciate them
    int totalCells = (partialRow - 1) * (partialCol - 1);

    double* subMatrix = (double*)malloc(sizeof(double) * totalCells );

    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(5, 1, 1);

    errCheck(cudaSetDevice(0));

    errCheck(cudaMalloc((void**)&dev_subMatrix, totalCells * sizeof(int)));
    errCheck(cudaMalloc((void**)&dev_matrix, totalCells * sizeof(int)));
    //errCheck(cudaMalloc((void**)&dev_b, totalCells * sizeof(int)));

    //errCheck(cudaMemcpy(dev_matrix, matrix, totalCells * sizeof(int), cudaMemcpyHostToDevice));
    errCheck(cudaMemcpy(dev_matrix, matrix, totalCells * sizeof(int), cudaMemcpyHostToDevice));
    //errCheck(cudaMemcpy(dev_b, b, totalCells * sizeof(int), cudaMemcpyHostToDevice));
    
    // for should be here
    // extract partial matrix from total matrix

    // in our addKernel, arguments should be
    // totalInputMatrix, partialMatrix, totalOutputMatrix
    //addKernel << <dimGrid, dimBlock >> > (dev_matrix, dev_b, dev_subMatrix, totalCells);

    // Kernel invocation
    //dim3 threadsPerBlock(16, 16);
    //dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    //MatAdd << <numBlocks, threadsPerBlock >> > (A, B, C);
    addKernel <<<dimGrid, dimBlock>>> (dev_matrix, dev_subMatrix, totalCells);

    errCheck(cudaGetLastError());
    errCheck(cudaDeviceSynchronize());
    errCheck(cudaMemcpy(subMatrix, dev_subMatrix, totalCells * sizeof(int), cudaMemcpyDeviceToHost));
    errCheck(cudaFree(dev_matrix));
    //errCheck(cudaFree(dev_b));
    errCheck(cudaFree(dev_subMatrix));
    errCheck(cudaDeviceReset());

    cout << "c = { " << subMatrix[0] << flush;
    for (int i = 1; i < totalCells; i++) {
        cout << ", " << subMatrix[i] << flush;
    }

    cout << " }" << endl << flush;

    
}

void checkLocalDevice() {
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#occupancy-calculator

    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        addKernel,
        blockSize,
        0);

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Max threads(wrap) that can be run at once: " << prop.warpSize << std::endl;
    std::cout << "Max dimension size of a thread block (x,y,z): " << "(" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max dimension size of a grid size    (x,y,z):: " << "(" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
}

// todo
double* convert2DMatTo1D(double** matrix) {
    double* convertedMatrix = nullptr;

    return convertedMatrix;
}