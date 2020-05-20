#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

#define ROWS 8
#define COLS 8
#define DATA_LENGTH 10

int ** allocateMatrix(int rows, int cols) {
    int ** matrix = (int **) malloc(rows * sizeof(int *));

    for(int i = 0; i < rows; i++) {
        matrix[i] = (int *) malloc(cols * sizeof(int));
    }

    return matrix;
}

void deallocateMatrix(int rows, int ** matrix) {
    for(int i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix);
}

void fillMatrix(int rows, int cols, int initialValue, int ** matrix) {
     for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            matrix[i][j] = initialValue;
        }
    }
}

void printMatrix(int rows, int cols, int ** matrix) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%12d ", matrix[i][j]);
        }

        printf("\n");
    }

    printf("\n");
}

void printRuntime(struct timeval tvs, struct timeval tve) {
    long start = tvs.tv_sec * 1000000 + tvs.tv_usec;
    long end = tve.tv_sec * 1000000 + tve.tv_usec;
    long delta = end - start;
    printf("Runtime: %.6f seconds\n", delta / 1000000.0);
}

void solveFirst(int rows, int cols, int iterations, struct timespec ts_sleep, int ** matrix) {
    // for(int k = 1; k <= iterations; k++) {
    //     for(int j = 0; j < cols; j++) {
    //         for(int i = 0; i < rows; i++) {
    //             usleep(1000);
    //             matrix[i][j] = matrix[i][j] + (i + j) * k;
    //         }
    //     }
    // }
}

void solveSecond(int rows, int cols, int iterations, struct timespec ts_sleep, int ** matrix) {
    // for(int k = 1; k <= iterations; k++) {
    //     for(int i = 0; i < rows; i++) {
    //         usleep(1000);
    //         matrix[i][0] = matrix[i][0] + (i * k);
    //     }

    //     for(int j = 1; j < cols; j++) {
    //         for(int i = 0; i < rows; i++) {
    //             usleep(1000);
    //             matrix[i][j] = matrix[i][j] + matrix[i][j - 1] * k;
    //         }
    //     }
    // }
}

void (* solve)(int rows, int cols, int iterations, struct timespec ts_sleep, int ** matrix) = solveFirst;

int main(int argc, char* argv[]) {

    // validate and ensure 3 args received
    if(4 != argc) {
        printf("[Err] : 3 args needed. Ex: lab1 2 0 2\n");
        printf("1st arg : Problem # to execute (1 or 2)\n");
        printf("2nd arg : initial initialValues to set in Matrix\n");
        printf("3rd arg : # of iterations\n");
        return EXIT_FAILURE;
    }

    // Parameter initialization from arguments
    int problemChoice = atoi(argv[1]);
    int initialValue = atoi(argv[2]);
    int numberOfIterations = atoi(argv[3]);

    struct timeval timestamp_s;
    struct timeval timestamp_e;
    struct timespec ts_sleep;
    ts_sleep.tv_sec = 0;
    ts_sleep.tv_nsec = 1000000L;

    // initialize MPI environment
    // argc : pointer to number of arguments
    // argc : pointer to the argument vector
    // can only be called by ONE thread!
    int errorState;
    errorState = MPI_Init(&argc, &argv);

    if (errorState != MPI_SUCCESS){
        printf("[Err] : Problem with Initializing MPI.\n");
        return EXIT_FAILURE;
    }

    // initialize solvers
    void * solvers[2];
    solvers[0] = solveFirst;
    solvers[1] = solveSecond;
    // set solver to Problem #1 or #2 based on arguments received
    solve = solvers[problemChoice - 1];

    // initialize Matrix
    int ** matrix = allocateMatrix(ROWS, COLS);
    fillMatrix(ROWS, COLS, initialValue, matrix);

    solve(ROWS, COLS, numberOfIterations, ts_sleep, matrix);
   
    // Finalize MPI environment
    MPI_Finalize();
    return EXIT_SUCCESS;
}
