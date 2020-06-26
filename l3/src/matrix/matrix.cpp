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

void debug_fillMatrixWithSeed(int rows, int cols, float seed, double ** matrix) {
     for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            matrix[row][col] = row * (rows - row - 1) * col * (cols - col - 1) * seed;
        }
    }
}

bool cloneMatValuesAtoB(int rowsA, int colsA, double ** matrixA, int rowsB, int colsB, double ** matrixB){
    // clones matrix values from A to B, returns true if successful, false if there's an error.

    return true;
}

bool isMatEqual(int rowsA, int colsA, double ** matrixA, int rowsB, int colsB, double ** matrixB){
    // compares two matrixes and returns true is they have matching values, false if not.
    return true;
}

double ** allocatePartialMatFromTargetMat(int * pmRows, int * pmCols, double ** partialMatrix, int tmRows, int tmCols, double ** targetMatrix){
    // allocate, initiazes and returns a partialMatrix. the partialMatrix's # number of rows and cols are set based on its targetMatrix
    // must adapt to various sizes

    return partialMatrix;
}

bool mirrorPartialMatToTargetMat(int pmRows, int pmCols, double ** partialMatrix, int tmRows, int tmCols, double ** targetMatrix){
    // takes a partial matrix and mirrors it to the remaining 3 quadrants of the targetMatrix. returns true if successful.
    // must adapt to various sizes
    return true;
}


