#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "parallel.cuh"

//#define ELEMENTS 5
#define MAXTHREADSPERBLOCK 1024

#define errCheck(code) { errorCheck((code), __FILE__, __LINE__); }
void addWithCuda(int rows, int cols, int iterations, double td, double h, double** matrix);
double* convert2DMatTo1D(int rows, int cols, double** matrix);
void transferToTargetMatrix(int rows, int cols, double* sourceMatrix, double** targetMatrix);
void checkLocalDevice();

using std::cout;
using std::flush;
using std::endl;
using std::fixed;
using std::setprecision;
using std::setw;

inline void errorCheck(cudaError_t code, const char* file, int line) {
    if (cudaSuccess != code) {
        std::cout << "[" << file << ", line " << line << "]" << std::flush;
        std::cout << " CUDA error <" << cudaGetErrorString(code) << "> received." << std::endl << std::flush;
        exit(EXIT_FAILURE);
    }
}

__global__ void addKernel(int rows, int cols, double interations, double td, double h, double* matrix, int elements) {
    extern __shared__ int threadCount;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //printf("x = %d, BlockIdx = %d, BlockDimx = %d, threadIdx = %d\n", x, blockIdx.x, blockDim.x, threadIdx.x);

    if (((x < (cols - 1)) && (x > 0)) &&
        ((y < (rows - 1)) && (y > 0))) {

        printf("x = %d, BlockIdx = %d, BlockDimx = %d, threadIdx = %d\n", x, blockIdx.x, blockDim.x, threadIdx.x);
        printf("y = %d, BlockIdy = %d, BlockDimy = %d, threadIdy = %d\n", y, blockIdx.y, blockDim.y, threadIdx.y);
        threadCount++;
        printf("ThreadCount = %i\n", threadCount);
    }
}

void solvePar(int rows, int cols, int iterations, double td, double h, double** matrix) {


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

    // min possible rows/cols/iterations = 3
    // max possible rows/cols/iterations = 10 000

    double* dev_matrix = nullptr;
    double* convertedMatrix = nullptr;


    convertedMatrix = convert2DMatTo1D(rows, cols, matrix);


    // calculate the total number of tiles to process, instanciate them
    int totalCells = rows * cols;

    // we will max out all possible threads in a block (so 1024)
    dim3 dimensionGrid(32, 32);
    
    // depending on our matrix size, our blocks will adjust
    // so a matrix of 12x12, max threads = 144, 1x1 block
    // matrix of 10000x10000, max threads = 100 000 000, 313x313 blocks
    double xBlocks = ceil(double(cols) / double(dimensionGrid.x));
    double yBlocks = ceil(double(rows) / double(dimensionGrid.y));

    dim3 dimensionBlock(xBlocks, yBlocks);

    std::cout << "Threads/block =  " << dimensionGrid.x* dimensionGrid.y << std::endl;
    std::cout << "DimBlocks [x,y] =  " << xBlocks << ", " << yBlocks << std::endl;
    

    errCheck(cudaSetDevice(0));
    errCheck(cudaMalloc((void**)&dev_matrix, totalCells * sizeof(double)));
    errCheck(cudaMemcpy(dev_matrix, convertedMatrix, totalCells * sizeof(double), cudaMemcpyHostToDevice));
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
    addKernel <<<dimensionBlock, dimensionGrid >>> (rows, cols, iterations, td, h, dev_matrix, totalCells);

    errCheck(cudaGetLastError());
    errCheck(cudaDeviceSynchronize());
    errCheck(cudaMemcpy(convertedMatrix, dev_matrix, totalCells * sizeof(int), cudaMemcpyDeviceToHost));
    errCheck(cudaFree(dev_matrix));
    //errCheck(cudaFree(dev_b));

    errCheck(cudaDeviceReset());

    // preview our matrix before transfer
    for (int i = 0; i < rows; i++) {
        cout << "{" << flush;
        for (int j = 0; j < cols; j++) {
            std::cout << std::fixed << std::setw(12) << std::setprecision(2) << convertedMatrix[i * cols + j] << " " << std::flush;
        }
        cout << " }" << endl << flush;
    }


    transferToTargetMatrix(rows, cols, convertedMatrix, matrix);
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


void transferToTargetMatrix(int rows, int cols, double* sourceMatrix, double** targetMatrix) {

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            targetMatrix[i][j] = sourceMatrix[i * cols + j];
        }
    }

}

double* convert2DMatTo1D(int rows, int cols, double** matrix) {
    double* convertedMatrix = new double[rows * cols];

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            convertedMatrix[i * cols +j] = matrix[i][j];
        }
    }

    return convertedMatrix;
}