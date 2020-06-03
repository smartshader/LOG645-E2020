#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

#define ROWS 8
#define COLS 8
#define MATRIX_LEN 10
#define MASTER_THREAD 0

struct position{
    int i_position;
    int j_position;
}coord[ROWS*COLS];

int *initMatrix(int matrixSize, int initialValue) {
    int *matrix = (int *) malloc(sizeof(int) * matrixSize);

    int rowIdentifier = 0;

    for (int matrixRowIterator = 0; matrixRowIterator < matrixSize; matrixRowIterator+=MATRIX_LEN){

        for (int rowIterator = 0; rowIterator < MATRIX_LEN; rowIterator++){
            if (rowIterator == 0){
                matrix[matrixRowIterator] = initialValue;
            }
            else if (rowIterator == 1){
                matrix[matrixRowIterator + 1] = rowIdentifier;
            }
            else{
                matrix[matrixRowIterator + rowIterator] = initialValue;
            }
        }
        rowIdentifier++;
    }
    return matrix;
}

void printMatrix(int *matrix, int matrixSize) {
    int colID = 0;

    for (int rowID = 0; rowID < matrixSize; rowID+=MATRIX_LEN){
        printf("%10d ", matrix[rowID]);
        colID++;
        if ((colID % COLS) == 0 && rowID != 0){
            colID = 0;
            printf("\n");
        }
    }
    printf("\n");
}

void initializeCoordinates(){
    int cellId = 0;
    for (int i = 0; i < ROWS; ++i){
        for (int j = 0; j < COLS; ++j){
            coord[cellId].i_position = i;
            coord[cellId].j_position = j;
            ++cellId;
        }
    }
}

void printRuntime(struct timeval tvs, struct timeval tve) {
    long start = tvs.tv_sec * 1000000 + tvs.tv_usec;
    long end = tve.tv_sec * 1000000 + tve.tv_usec;
    long delta = end - start;
    printf("Runtime: %.6f seconds\n", delta / 1000000.0);
}

void printFinalResults(struct timeval timestamp_s, struct timeval timestamp_e, int *matrix, int matrixSize){
    printf("END - Master Thread\n");
    gettimeofday(&timestamp_e, NULL);
    printMatrix(matrix, matrixSize);
    printRuntime(timestamp_s, timestamp_e);
}

int *solveFirst(int *matrix, int cellsToProcess, int iteration)
{
    int matrixRowIterator, k;

    for (k = 0; k <= iteration; k++)
    {
        for (matrixRowIterator = 0; matrixRowIterator < cellsToProcess; matrixRowIterator += MATRIX_LEN)
        {
            usleep(1000);
            int iCurrent = coord[matrix[matrixRowIterator + 1]].i_position;
            int jCurrent = coord[matrix[matrixRowIterator + 1]].j_position;
            matrix[matrixRowIterator] = matrix[matrixRowIterator] + (iCurrent + jCurrent) * k;
        }
    }
    return matrix;
}

int *solveSecond(int *matrix, int cellsToProcess, int iteration) {

    return matrix;
}


int main(int argc, char* argv[]) {

    if(4 != argc) {
        printf("[Err] : 3 args needed. Ex: lab1 2 0 2\n");
        return EXIT_FAILURE;
    }

    // Parameter initialization
    int problemChoice = atoi(argv[1]);
    int initialValue = atoi(argv[2]);
    int iterations = atoi(argv[3]);

    struct timeval timestamp_s;
    struct timeval timestamp_e;

    // ---------------initialize MPI environment
    MPI_Init(&argc, &argv);

    gettimeofday(&timestamp_s, NULL);
    int instanceSize, cpuRank;
    MPI_Comm_size(MPI_COMM_WORLD, &instanceSize);

    if ((ROWS*COLS) % instanceSize != 0){
        printf("Multiple of 64 is required.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);
    
    int cellId = 0;
    for (int i = 0; i < ROWS; ++i){
        for (int j = 0; j < COLS; ++j){
            coord[cellId].i_position = i;
            coord[cellId].j_position = j;
            ++cellId;
        }
    }
    
    // initialize Matrix on root process
    int matrixSize = ROWS * COLS * MATRIX_LEN;
    int *wholeMatrix = NULL;

    if (cpuRank == MASTER_THREAD){
        wholeMatrix = initMatrix(matrixSize, initialValue);
    }

    // For each process, create subMatrix buffers based on cellsPerSubMatrix
    int cellsPerSubMatrix = (matrixSize) / instanceSize;
    int *subMatrix = (int *)malloc(sizeof(int) * cellsPerSubMatrix);

    // SCATTER STAGE - scatters from root proc to all process
    int scatterStatus = MPI_Scatter(wholeMatrix, 
                                    cellsPerSubMatrix, 
                                    MPI_INT, 
                                    subMatrix, 
                                    cellsPerSubMatrix, 
                                    MPI_INT, 
                                    0, 
                                    MPI_COMM_WORLD);

    if (scatterStatus != MPI_SUCCESS){
        printf("[Error] MPI_Scatter\n");
        return EXIT_FAILURE;
    }

    // COMPUTE STAGE
    int *singleSubMatrixCalculated = NULL;

    if (problemChoice == 1){
        singleSubMatrixCalculated = solveFirst(subMatrix, cellsPerSubMatrix, iterations);
    }
    else if (problemChoice == 2){
        singleSubMatrixCalculated = solveSecond(subMatrix, cellsPerSubMatrix, iterations);
    }
    else{
        printf("[Error] Select valid problem choice\n");
        return EXIT_FAILURE;
    }

    // GATHER STAGE - gathers all singleSubMatrixCalculated to the root process
    int *combinedSubMatrixes = NULL;

    if (cpuRank == MASTER_THREAD) {
        combinedSubMatrixes = (int *)malloc(sizeof(int) * matrixSize);
    }

    int data_gather = MPI_Gather(singleSubMatrixCalculated, 
                                    cellsPerSubMatrix, 
                                    MPI_INT, 
                                    combinedSubMatrixes, 
                                    cellsPerSubMatrix, 
                                    MPI_INT, 
                                    0, 
                                    MPI_COMM_WORLD);

    if (data_gather != MPI_SUCCESS){
        printf("[Error] MPI_Gather\n");
        return EXIT_FAILURE;
    }

    if (cpuRank == MASTER_THREAD) {
        printFinalResults(timestamp_s,timestamp_e, combinedSubMatrixes, matrixSize);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
