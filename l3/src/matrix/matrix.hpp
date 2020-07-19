#ifndef MATRIX_HPP
#define MATRIX_HPP

double ** allocateMatrix(int rows, int cols);

void deallocateMatrix(int rows, double ** matrix);

void fillMatrix(int rows, int cols, double ** matrix);

void debug_fillMatrixWithSeed(int rows, int cols, float seed, double ** matrix);



#endif
