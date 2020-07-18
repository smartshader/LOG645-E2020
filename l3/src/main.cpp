#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

#include <mpi.h>

#include "matrix/matrix.hpp"
#include "output/output.hpp"
#include "solver/solver.hpp"

void invalidArguments();

void printInputArguments(int argc, char* argv[]);
void initialMatrixDisplayOnly(int rows, int cols);

long sequential(int rows, int cols, int iters, double td, double h, int sleep, double ** tempSeqMatrix);
long parallel(int rows, int cols, int iters, double td, double h, int sleep, double ** tempParMatrix);

using namespace std::chrono;

using std::cout;
using std::endl;
using std::flush;
using std::setprecision;
using std::setw;
using std::stod;
using std::stoi;

int main(int argc, char* argv[]) {
    // Arguments.
    int rows;
    int cols;
    int iters;
    double td;
    double h;
    bool debugMode;

    // MPI variables.
    int mpi_status;
    int rank;

    // Resolution variables.
    // Sleep will be in microseconds during execution.
    int sleep = 1;

    // Timing variables.
    long runtime_seq = 0;
    long runtime_par = 0;

    // matrix placeholders for comparison
    double ** tempSeqMatrix = NULL;
    double ** tempParMatrix = NULL;

    // at the minimum, we need *at least* 5 arguments
    // ________________ MANDATORY (for submission)
    // 1: (int)    n - number of lines
    // 2: (int)    m - number of columns
    // 3: (int)    tp - number of timesteps/iterations
    // 4: (double) td - discretized time
    // 5: (float)  h - size of each tile subdivison (square hxh)

    // ________________ OPTIONAL (used for dev purposes)
    // 6: (bool) enable/disables matrix output
    if(argc < 5) {
        invalidArguments();
        return EXIT_FAILURE;
    }

    mpi_status = MPI_Init(&argc, &argv);
    
    if(MPI_SUCCESS != mpi_status) {
        cout << "MPI initialization failure." << endl << flush;
        return EXIT_FAILURE;
    }

    rows = stoi(argv[1], nullptr, 10);
    cols = stoi(argv[2], nullptr, 10);
    iters = stoi(argv[3], nullptr, 10);
    td = stod(argv[4], nullptr);
    h = stod(argv[5], nullptr);
    debugMode = stod(argv[6], nullptr);

    // matrix placeholders for comparison
    tempSeqMatrix = allocateMatrix(rows, cols);
    tempParMatrix = allocateMatrix(rows,cols);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(0 == rank) {
        printInputArguments(argc, argv);
        initialMatrixDisplayOnly(rows, cols);
        runtime_seq = sequential(rows, cols, iters, td, h, sleep, tempSeqMatrix);
    }

    // Ensure that no process will start computing early.
    MPI_Barrier(MPI_COMM_WORLD);

    // TODO howard : maybe partial matrix should be initialized here instead?

    runtime_par = parallel(rows, cols, iters, td, h, sleep, tempParMatrix);

    // _________________ FINAL RESULTS
    if(0 == rank) {
        printStatistics(1, runtime_seq, runtime_par);

        deallocateMatrix(rows, tempSeqMatrix);
        deallocateMatrix(rows, tempParMatrix);
    }

    mpi_status = MPI_Finalize();
    if(MPI_SUCCESS != mpi_status) {
        cout << "Execution finalization terminated in error." << endl << flush;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// =======================================================================================

void invalidArguments() {
    cout << "Invalid arguments." << endl << flush;
    cout << "Arguments: m n np td h" << endl << flush;
}

void printInputArguments(int argc, char* argv[]) {
    cout << "Configuration:" << flush;

    for(int i = 0; i < argc; i++) {
        cout << " " << argv[i] << flush;
    }

    cout << endl << flush;
}

void initialMatrixDisplayOnly(int rows, int cols) {
    double ** matrix = allocateMatrix(rows, cols);

    fillMatrix(rows, cols, matrix);
    cout << "-----  INITIAL   -----" << endl << flush;
    printMatrix(rows, cols, matrix);
    deallocateMatrix(rows, matrix);
}

long sequential(int rows, int cols, int iters, double td, double h, int sleep, double ** tempSeqMatrix) {
    double ** matrix = allocateMatrix(rows, cols);
    fillMatrix(rows, cols, matrix);

    time_point<high_resolution_clock> timepoint_s = high_resolution_clock::now();
    solveSeq(rows, cols, iters, td, h, sleep, matrix);
    time_point<high_resolution_clock> timepoint_e = high_resolution_clock::now();

    cout << "----- SEQUENTIAL -----" << endl << flush;
    printMatrix(rows, cols, matrix);

    deallocateMatrix(rows, matrix);
    return duration_cast<microseconds>(timepoint_e - timepoint_s).count();
}

long parallel(int rows, int cols, int iters, double td, double h, int sleep, double ** tempParMatrix) {
    double ** matrix = allocateMatrix(rows, cols);
    fillMatrix(rows, cols, matrix);
    // int pmRows, pmCols;
    // double ** partialMatrix;

    // partialMatrix = allocatePartialMatFromTargetMat(&pmRows, &pmCols, rows,cols,matrix);
    // TODO Howard : maybe partial matrix should be generated here ?

    time_point<high_resolution_clock> timepoint_s = high_resolution_clock::now();
    solvePar(rows, cols, iters, td, h, sleep, matrix);
    time_point<high_resolution_clock> timepoint_e = high_resolution_clock::now();

    if(nullptr != *matrix) {
        cout << "-----  PARALLEL  -----" << endl << flush;
        printMatrix(rows, cols, matrix);
        deallocateMatrix(rows, matrix);
    }

    return duration_cast<microseconds>(timepoint_e - timepoint_s).count();
}

