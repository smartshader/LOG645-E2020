#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

#include <mpi.h>

#include "matrix/matrix.hpp"
#include "output/output.hpp"
#include "solver/solver.hpp"

#define MASTER_CPU 0
// color
#define GREEN   "\033[32m"     
#define YELLOW   "\033[33m"  
#define RESET   "\033[0m"

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

    // Resolution variables.
    // Sleep will be in microseconds during execution.
    int sleep = 1;

    // Timing variables.
    long runtime_seq = 0;
    long runtime_par = 0;

    // temporary matrix storage use to store and compare final answers
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

    // initialize arguments
    rows = stoi(argv[1], nullptr, 10);
    cols = stoi(argv[2], nullptr, 10);
    iters = stoi(argv[3], nullptr, 10);
    td = stod(argv[4], nullptr);
    h = stod(argv[5], nullptr);
    debugMode = stod(argv[6], nullptr);

    tempSeqMatrix = allocateMatrix(rows, cols);
    tempParMatrix = allocateMatrix(rows, cols);

    // create an instance of MPI World
    int mpi_status;
    mpi_status = MPI_Init(&argc, &argv);
    
    if(MPI_SUCCESS != mpi_status) {
        cout << "MPI initialization failure." << endl << flush;
        return EXIT_FAILURE;
    }

    // get the current rank of CPU
    int cpuRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);

    // ___________________________________________________ Sequential
    if(MASTER_CPU == cpuRank) {
        printInputArguments(argc, argv);
        initialMatrixDisplayOnly(rows, cols);
        runtime_seq = sequential(rows, cols, iters, td, h, sleep, tempSeqMatrix);
    }

    // Ensure that no process will start computing early.
    MPI_Barrier(MPI_COMM_WORLD);
    // ___________________________________________________ Parallel
    runtime_par = parallel(rows, cols, iters, td, h, sleep, tempParMatrix);

    // ___________________________________________________ RESULTS
    if(MASTER_CPU == cpuRank) {
        printStatistics(1, runtime_seq, runtime_par);
        // only need a single row to deallocate everything
        deallocateMatrix(rows, tempSeqMatrix);
        deallocateMatrix(rows, tempParMatrix);
    }

    mpi_status = MPI_Finalize();

    if(mpi_status != MPI_SUCCESS) {
        cout << "Execution finalization terminated in error." << endl << flush;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// =======================================================================================


long parallel(int rows, int cols, int iters, double td, double h, int sleep, double ** tempParMatrix) {
    // partialMatrix dimensions
    int pmRows, pmCols;

    double ** targetMatrix = NULL;
    double ** partialMatrix = NULL;

    targetMatrix = allocateMatrix(rows, cols);
    fillMatrix(rows, cols, targetMatrix);
    partialMatrix = allocatePartialMatFromTargetMat(&pmRows, &pmCols, rows, cols, targetMatrix);

    time_point<high_resolution_clock> timepoint_s = high_resolution_clock::now();
    // partialMatrix injection should be here...
    solvePar(rows, cols, iters, td, h, sleep, targetMatrix);
    time_point<high_resolution_clock> timepoint_e = high_resolution_clock::now();


    // recall that solvePar deallocates everything except the MASTER_CPU
    // only our MASTER_CPU (Rank 0) will output the final result
    if(*targetMatrix != nullptr) {
        cout << YELLOW << "---------------------- partialMatrix -------- " << RESET << endl << flush;
        printMatrix(pmRows, pmCols, partialMatrix);
        
        cout << "-----  PARALLEL RES -----" << endl << flush;

        // TODO : copy targetMatrix results to tempParMatrix
        printMatrix(rows, cols, targetMatrix);
        deallocateMatrix(rows, targetMatrix);
        deallocateMatrix(pmRows, partialMatrix);
    }

    return duration_cast<microseconds>(timepoint_e - timepoint_s).count();
}










long sequential(int rows, int cols, int iters, double td, double h, int sleep, double ** tempSeqMatrix) {
    double ** targetMatrix = allocateMatrix(rows, cols);
    fillMatrix(rows, cols, targetMatrix);

    time_point<high_resolution_clock> timepoint_s = high_resolution_clock::now();
    solveSeq(rows, cols, iters, td, h, sleep, targetMatrix);
    time_point<high_resolution_clock> timepoint_e = high_resolution_clock::now();

    cout << "----- SEQUENTIAL RES -----" << endl << flush;
    // TODO : copy matrix results
    printMatrix(rows, cols, targetMatrix);
    deallocateMatrix(rows, targetMatrix);
    return duration_cast<microseconds>(timepoint_e - timepoint_s).count();
}


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
    cout << "-----  INITIAL MATRIX   -----" << endl << flush;
    printMatrix(rows, cols, matrix);
    deallocateMatrix(rows, matrix);
}