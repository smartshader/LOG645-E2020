#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "parallel.cuh"

//#define ELEMENTS 5

#define errCheck(code) { errorCheck((code), __FILE__, __LINE__); }
void addWithCuda(int rows, int cols, int iterations, double td, double h, double** matrix);
double* convert2DMatTo1D(int rows, int cols, double** matrix);
void transferToTargetMatrix(int rows, int cols, double* sourceMatrix, double** targetMatrix);
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

__global__ void addKernel(int rows, int cols, double* matrix, int elements) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    printf("x = %d, BlockIdx = %d, BlockDimx = %d, threadIdx = %d\n", x, blockIdx.x, blockDim.x, threadIdx.x);
    if (x < elements) {
        //c[x] = a[x] + b[x];
        matrix[x] = matrix[x];
        //matrix[i][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
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

    // min possible rows/cols/iterations = 3
    // max possible rows/cols/iterations = 10 000

    double* dev_matrix = nullptr;
    double* convertedMatrix = nullptr;


    convertedMatrix = convert2DMatTo1D(rows, cols, matrix);


    // calculate the total number of tiles to process, instanciate them
    int totalCells = rows * cols;


    // 16 by 16 threads for each block (in my case I can only have 32 threads at once)
    // something to note to keep dynamic thread allocation as per device
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(cols/threadsPerBlock.x, rows/threadsPerBlock.y);
    

    errCheck(cudaSetDevice(0));


    errCheck(cudaMalloc((void**)&dev_matrix, totalCells * sizeof(double)));
    //errCheck(cudaMalloc((void**)&dev_b, totalCells * sizeof(int)));

    //errCheck(cudaMemcpy(dev_matrix, matrix, totalCells * sizeof(int), cudaMemcpyHostToDevice));
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
    addKernel <<<numBlocks, threadsPerBlock>>> (rows, cols, dev_matrix, totalCells);

    errCheck(cudaGetLastError());
    errCheck(cudaDeviceSynchronize());
    errCheck(cudaMemcpy(convertedMatrix, dev_matrix, totalCells * sizeof(int), cudaMemcpyDeviceToHost));
    errCheck(cudaFree(dev_matrix));
    //errCheck(cudaFree(dev_b));

    errCheck(cudaDeviceReset());


    for (int i = 1; i < totalCells; i++) {
        cout << ", " << convertedMatrix[i] << flush;
    }

    cout << " }" << endl << flush;

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