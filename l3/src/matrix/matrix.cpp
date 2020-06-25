#include "matrix.hpp"

double ** allocateMatrix(int rows, int cols) {
    double ** matrix = new double*[rows];

    for(int i = 0; i < rows; i++) {
        matrix[i] = new double[cols];
    }

    return matrix;
}

void deallocateMatrix(int rows, double ** matrix) {
    for(int i = 0; i < rows; i++) {
        delete(matrix[i]);
        matrix[i] = nullptr;
    }

    delete(matrix);
    *matrix = nullptr;
}

void fillMatrix(int rows, int cols, double ** matrix) {
     for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            matrix[row][col] = row * (rows - row - 1) * col * (cols - col - 1);
        }
    }
}

void fillMatrixWithSeed(int rows, int cols, float seed, double ** matrix) {
     for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            matrix[row][col] = row * (rows - row - 1) * col * (cols - col - 1) * seed;
        }
    }
}

bool cloneMatValuesAtoB(double ** matrixB, double ** matrixA, int rows, int cols){
    // clones matrix values from A to B, returns true if successful, false if there's an error.

    return true;
}

bool isMatEqual(double ** matrixA, double ** matrixB, int rows, int cols){
    // compares two matrixes and returns true is they have matching values, false if not.
    return true;
}

double ** allocatePartialMatFromTargetMat(double ** partialMatrix, int * pmRows, int * pmCols, double ** targetMatrix, int tmRows, int tmCols){
    // allocate and initiazes a Partial Matrix, its number of rows and cols from a Target Matrix
    // must adapt to various sizes

    return partialMatrix;
}

bool mirrorPartialMatToTargetMat(double ** partialMatrix, int pmRows, int pmCols, double ** targetMatrix, int tmRows, int tmCols){
    // takes a partial matrix and mirrors it to the remaining 3 quadrants. returns true if successful.
    // must adapt to various sizes
    return true;
}


