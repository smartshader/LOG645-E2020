#include <iomanip>
#include <iostream>
#include <chrono>

#include "output.hpp"
#include "../matrix/matrix.hpp"

using namespace std::chrono;

using std::cout;
using std::endl;
using std::fixed;
using std::flush;
using std::setprecision;
using std::setw;

void printMatrix(int rows, int cols, double ** matrix) {
    for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            cout << fixed << setw(4) << setprecision(0) << matrix[row][col] << flush;
        }

        cout << endl << flush;
    }

    cout << endl << flush;
}

void printStatistics(int threads, long runtime_seq, long runtime_par) {
    double acceleration = 1.0 * runtime_seq / runtime_par;
    double efficiency = acceleration / threads;

    cout << "Runtime sequential: " << runtime_seq / 1000000.0 << " seconds" << endl << flush;
    cout << "Runtime parallel  : " << runtime_par / 1000000.0 << " seconds" << endl << flush;
    cout << "Acceleration      : " << acceleration << endl << flush;
    cout << "Efficiency        : " << efficiency << endl << flush;
}

void debug_printStatistics(int threads, long runtime_seq, long runtime_par,double ** matrixA, double ** matrixB) {
    double acceleration = 1.0 * runtime_seq / runtime_par;
    double efficiency = acceleration / threads;

    cout << "Runtime sequential: " << runtime_seq / 1000000.0 << " seconds" << endl << flush;
    cout << "Runtime parallel  : " << runtime_par / 1000000.0 << " seconds" << endl << flush;
    cout << "Acceleration      : " << acceleration << endl << flush;
    cout << "Efficiency        : " << efficiency << endl << flush;
}

void debug_isMatEqual(int rows, int cols, double ** matrixA, double ** matrixB){
    // DEBUG -------------------------------------------------------- compares two EQUAL matrixes
    fillMatrix(rows, cols, matrixA);
    cout << "Matrix Seq (true test) : tempSeqMatrix" << endl << flush;
    printMatrix(rows, cols, matrixA);
    fillMatrix(rows, cols, matrixB);
    cout << "Matrix Par (true test) : tempParMatrix" << endl << flush;
    printMatrix(rows, cols, matrixB);

    bool isEqual = isMatEqual(rows, cols, matrixA, matrixB);

    if(isEqual == true){
        printf("Matrix A and B are equal\n");		
    }
    else{
        printf("Matrix A and B are different\n");			
    }

    // DEBUG -------------------------------------------------------- compares two NON-EQUAL matrixes
    debug_fillMatrixWithSeed(rows, cols, 2.5, matrixA);
    cout << "Matrix Seq (false test) : tempSeqMatrix" << endl << flush;
    printMatrix(rows, cols, matrixA);
    debug_fillMatrixWithSeed(rows, cols, 4.3, matrixB);
    cout << "Matrix Par (false test) : tempParMatrix" << endl << flush;
    printMatrix(rows, cols, matrixB);

    isEqual = isMatEqual(rows, cols, matrixA, matrixB);

    if(isEqual == true){
        printf("Matrix A and B are equal\n");		
    }
    else{
        printf("Matrix A and B are different\n");			
    }
}
