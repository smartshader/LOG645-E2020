#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "parallel.cuh"

#define errCheck(code) { errorCheck((code), __FILE__, __LINE__); }
void addWithCuda(int rows, int cols, int iterations, double td, double h, double** matrix);
double* convert2DMatTo1D(int rows, int cols, double** matrix);
void transferToTargetMatrix(int rows, int cols, double* sourceMatrix, double** targetMatrix);

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

__global__ void addKernel(int rows, int cols, double td, double h, double* matrix) {
    extern __shared__ int threadCount;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (((x < (cols - 1)) && (x > 0)) &&
        ((y < (rows - 1)) && (y > 0))) {

        threadCount++;

        double h_area = h * h;
        double l = matrix[x * cols + (y - 1)];
        double r = matrix[x * cols + (y + 1)];
        double t = matrix[(x - 1) * cols + y];
        double b = matrix[(x + 1) * cols + y];
        double c = matrix[x * cols + y];
        matrix[x * cols + y] = (1.0 - 4.0 * td / h_area) * c + (td / h_area) * (t + b + l + r);
    }
}

void solvePar(int rows, int cols, int iterations, double td, double h, double** matrix) {

    addWithCuda(rows, cols, iterations, td, h, matrix);
}

void addWithCuda(int rows, int cols, int iterations, double td, double h, double** matrix) {

    // min possible rows/cols/iterations = 3
    // max possible rows/cols/iterations = 10 000

    double* dev_matrix = nullptr;
    double* convertedMatrix = nullptr;

    convertedMatrix = convert2DMatTo1D(rows, cols, matrix);

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
    errCheck(cudaMalloc((void**)&dev_matrix, dimensionGrid.x * dimensionGrid.y * sizeof(double)));
    errCheck(cudaMemcpy(dev_matrix, convertedMatrix, dimensionGrid.x * dimensionGrid.y *sizeof(double), cudaMemcpyHostToDevice));

    for (int k = 0; k < iterations; k++) {
        addKernel << <dimensionBlock, dimensionGrid >> > (rows, cols, td, h, dev_matrix);
    }

    errCheck(cudaGetLastError());
    errCheck(cudaDeviceSynchronize());
    errCheck(cudaMemcpy(convertedMatrix, dev_matrix, dimensionGrid.x * dimensionGrid.y * sizeof(int), cudaMemcpyDeviceToHost));
    errCheck(cudaFree(dev_matrix));
    errCheck(cudaDeviceReset());

    transferToTargetMatrix(rows, cols, convertedMatrix, matrix);
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

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            convertedMatrix[i * cols +j] = matrix[i][j];
        }
    }
    return convertedMatrix;
}