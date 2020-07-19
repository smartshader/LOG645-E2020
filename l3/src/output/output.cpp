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

#define GREEN "\033[32m"      
#define MAGENTA "\033[35m"    
#define RED  "\033[31m"  
#define RESET   "\033[0m"
#define MASTER_CPU 0

void printMatrix(int rows, int cols, double ** matrix) {
    for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            cout << fixed << setw(12) << setprecision(2) << matrix[row][col] << flush;
        }
        cout << endl << flush;
    }
    cout << endl << flush;
}

void printStatistics(int threads, long runtime_seq, long runtime_par) {
    double acceleration = 1.0 * runtime_seq / runtime_par;
    double efficiency = acceleration / threads;

    cout << "Runtime sequential: " << runtime_seq << " uS" << endl << flush;
    cout << "Runtime parallel  : " << runtime_par  << " uS" << endl << flush;
    cout << "Acceleration      : " << acceleration << endl << flush;
    cout << "Efficiency        : " << efficiency << endl << flush;
    cout << "# threads  : " << threads << endl << flush;
}

// used for measuring/truncated results
void debug_printStatistics(int threads, long runtime_seq, long runtime_par, int rows, int cols) {
    double acceleration = 1.0 * runtime_seq / runtime_par;
    double efficiency = acceleration / threads;
    cout << fixed << setw(6) << ": " << runtime_seq << " uS, "
    << fixed << setw(12) << runtime_par << " uS, "
    << fixed << setw(6) << setprecision(4) << acceleration << ", "
    << fixed << setw(6) << setprecision(4) << efficiency << ", "
    << fixed << setw(2) << threads << endl << flush;
}
