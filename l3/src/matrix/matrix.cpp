#include <thread>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "matrix.hpp"
#include "../output/output.hpp"

#define GREEN   "\033[32m"
#define RESET   "\033[0m"

using namespace std::chrono;

using std::cerr;
using std::endl;
using std::flush;
using std::cout;
using std::fixed;
using std::setprecision;
using std::setw;

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

// clones matrix values from A to B, returns true if successful, false if there's an error.
bool cloneMatValuesAtoB(int rows, int cols, double ** matrixA, double ** matrixB){
	try{
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < cols; col++) {
				matrixB[row][col] = matrixA[row][col];
			}
		}	
		return true;	
	}
	catch(...){
		return false;
	}
}

// compares two matrixes and returns true is they have matching values, false if not.
bool isMatEqual(int rows, int cols, double ** matrixA, double ** matrixB){
    
	for(int row =0; row < rows; row++){
		for(int col=0; col < cols; col++){
			if(!double_equals(matrixA[row][col],matrixB[row][col])) {
				return false;
			}
		}
	}
    return true;
}

// required when comparing double/floating points
bool double_equals(double a, double b)
{
	double epsilon = 0.001;
    return std::abs(a - b) < epsilon;
}

// fills a matrix with seed
void debug_fillMatrixWithSeed(int rows, int cols, float seed, double ** matrix) {
     for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            matrix[row][col] = row * (rows - row - 1) * col * (cols - col - 1) * seed;
        }
    }
}