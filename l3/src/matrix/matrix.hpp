#ifndef MATRIX_HPP
#define MATRIX_HPP

double ** allocateMatrix(int rows, int cols);

void deallocateMatrix(int rows, double ** matrix);
bool double_equals(double a, double b);

void fillMatrix(int rows, int cols, double ** matrix);

void debug_fillMatrixWithSeed(int rows, int cols, float seed, double ** matrix);

bool cloneMatValuesAtoB(int rows, int cols, double ** matrixA, double ** matrixB);

bool isMatEqual(int rows, int cols, double ** matrixA, double ** matrixB);

#endif
