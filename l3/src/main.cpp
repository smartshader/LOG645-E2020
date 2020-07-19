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

long sequential(int rows, int cols, int iters, double td, double h, int sleep, bool regularOutputs);
long parallel(int rows, int cols, int iters, double td, double h, int sleep, int cpuRank, bool regularOutputs);

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

    //// CHANGE THIS BEFORE SUBMISSION
    bool regularOutputs = false; //default is 0

    // Resolution variables.
    // Sleep will be in microseconds during execution.
    int sleep = 1;

    // Timing variables.
    long runtime_seq = 0;
    long runtime_par = 0;

    // at the minimum, we need *at least* 5 arguments
    // ________________ MANDATORY (for submission)
    // 1: (int)    n - number of lines
    // 2: (int)    m - number of columns
    // 3: (int)    tp - number of timesteps/iterations
    // 4: (double) td - discretized time
    // 5: (float)  h - size of each tile subdivison (square hxh)

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

    // initializes MPI space
    MPI_Init(&argc, &argv);

    // get the current rank of CPU
    int cpuRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);

    if(MASTER_CPU == cpuRank) {
        printInputArguments(argc, argv);
        if (regularOutputs) initialMatrixDisplayOnly(rows, cols);
        runtime_seq = sequential(rows, cols, iters, td, h, sleep, regularOutputs);
    }

    // Ensure that no process will start computing early.
    MPI_Barrier(MPI_COMM_WORLD);

    runtime_par = parallel(rows, cols, iters, td, h, sleep, cpuRank, regularOutputs);

    // ___________________________________________________ RESULTS
    if(MASTER_CPU == cpuRank) {

        // statistics
        int instanceSize;
        MPI_Comm_size(MPI_COMM_WORLD, &instanceSize);
        if (regularOutputs == false){
            debug_printStatistics(instanceSize, runtime_seq, runtime_par, rows, cols);
        }
        else
        {
            printStatistics(instanceSize, runtime_seq, runtime_par);
        }
    }

    // terminates MPI execution environment
    MPI_Finalize();
    return EXIT_SUCCESS;
}

long parallel(int rows, int cols, int iters, double td, double h, int sleep, int cpuRank, bool regularOutputs) {

    double ** targetMatrix = allocateMatrix(rows, cols);
    fillMatrix(rows, cols, targetMatrix);

    time_point<high_resolution_clock> timepoint_s = high_resolution_clock::now();
    solvePar(rows,cols,iters,td,h,sleep,targetMatrix);
    time_point<high_resolution_clock> timepoint_e = high_resolution_clock::now();

    if(*targetMatrix != nullptr) {
        if (cpuRank == 0){
            // debug purposes
            if (regularOutputs){
                cout << "-----  PARALLEL RES -----" << endl << flush;
                printMatrix(rows, cols, targetMatrix);
            }
        }
        deallocateMatrix(rows, targetMatrix);
    }
    return duration_cast<microseconds>(timepoint_e - timepoint_s).count();
}

long sequential(int rows, int cols, int iters, double td, double h, int sleep, bool regularOutputs) {
    double ** targetMatrix = allocateMatrix(rows, cols);
    fillMatrix(rows, cols, targetMatrix);

    time_point<high_resolution_clock> timepoint_s = high_resolution_clock::now();
    solveSeq(rows, cols, iters, td, h, sleep, targetMatrix);
    time_point<high_resolution_clock> timepoint_e = high_resolution_clock::now();

    // debug purposes
    if (regularOutputs){
        cout << "----- SEQUENTIAL RES -----" << endl << flush;
        printMatrix(rows, cols, targetMatrix);
    }

    deallocateMatrix(rows, targetMatrix);
    return duration_cast<microseconds>(timepoint_e - timepoint_s).count();
}

void invalidArguments() {
    cout << "Invalid arguments." << endl << flush;
    cout << "Arguments: m n np td h [debugMode]" << endl << flush;
}

void printInputArguments(int argc, char* argv[]) {
    cout << "Config:" << flush;

    for(int i = 0; i < argc; i++) 
        cout << " " << argv[i] << flush;
    cout << flush;
}

void initialMatrixDisplayOnly(int rows, int cols) {
    double ** matrix = allocateMatrix(rows, cols);
    fillMatrix(rows, cols, matrix);
    cout << "-----  INITIAL MATRIX   -----" << endl << flush;
    printMatrix(rows, cols, matrix);
    deallocateMatrix(rows, matrix);
}