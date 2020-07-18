#include <thread>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "matrix.hpp"
#include "../output/output.hpp"

// color color outputs
#define GREEN   "\033[32m"      /* Green */
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

void debug_fillMatrixWithSeed(int rows, int cols, float seed, double ** matrix) {
     for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            matrix[row][col] = row * (rows - row - 1) * col * (cols - col - 1) * seed;
        }
    }
}

bool cloneMatValuesAtoB(int rows, int cols, double ** matrixA, double ** matrixB){
    // clones matrix values from A to B, returns true if successful, false if there's an error.
	try{
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < cols; col++) {
				// cerr << "Row: " << row << "      Col: " << col << endl << flush;
				
				// cerr << "Value: " << matrixA[row][col] << endl << flush;				
				matrixB[row][col] = matrixA[row][col];
			}
		}	
		return true;	
	}
	catch(...){
		return false;
	}
}

bool isMatEqual(int rows, int cols, double ** matrixA, double ** matrixB){
    // compares two matrixes and returns true is they have matching values, false if not.
	for(int row =0; row < rows; row++){
		for(int col=0; col < cols; col++){
			if(matrixA[row][col] != matrixB[row][col]) {
				return false;
			}
		}
	}
    return true;
}

// allocate, initiazes and returns a partialMatrix, its # number of rows and cols are set based on its targetMatrix
// this function is called from solver.cpp
double ** allocatePartialMatFromTargetMat(int * pmRows, int * pmCols, int tmRows, int tmCols, double ** targetMatrix){
	
    // determine appropriate partialMatrix lengths
    *pmRows = tmRows/2 - ((tmRows % 2 == 0) ? 1 : 0);
    *pmCols = tmCols/2 - ((tmCols % 2 == 0) ? 1 : 0);

    // allocate it
    double ** partialMatrix = allocateMatrix(*pmRows, *pmCols);

    // fill our partial matrix
	for (int currentTargetRow = 1; currentTargetRow <= *pmRows; currentTargetRow++){
		for (int currentTargetCol = 1; currentTargetCol <= *pmCols; currentTargetCol++){
			partialMatrix[currentTargetRow - 1][currentTargetCol - 1] = targetMatrix[currentTargetRow][currentTargetCol];
		}
	}

    return partialMatrix;
}


bool mirrorPartialMatToTargetMat(int pmRows, int pmCols, double ** partialMatrix, int tmRows, int tmCols, double ** targetMatrix){
    // takes a partial matrix and mirrors it to the remaining 3 quadrants of the targetMatrix. returns true if successful.
    // must adapt to various sizes
	if(tmRows <= pmRows * 2 || tmCols <= pmCols * 2){
		return false;
	}
	for(int tmRow = 0; tmRow < tmRows; tmRow++) {
		for(int tmCol = 0; tmCol < tmCols; tmCol++) {
			if(tmRow == 0 ||  tmRow == tmRows - 1 || tmCol == 0 || tmCol == tmCols - 1){
				targetMatrix[tmRow][tmCol] = 0;
			}
			else{
				int pmRow, pmCol = 0;
				if(tmRow > pmRows){
					pmRow = tmRows - (tmRow + 2);
				}
				else{
					pmRow = tmRow - 1;
				}
				
				if(tmCol > pmCols){
					pmCol = tmCols - (tmCol + 2);
				}
				else{
					pmCol = tmCol - 1;
				}
				targetMatrix[tmRow][tmCol] = partialMatrix[pmRow][pmCol];
			}
		}
	}
    return true;
}

void restorePartialMatrix(){

}


