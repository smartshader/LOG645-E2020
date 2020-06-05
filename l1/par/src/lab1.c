#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

#define ROWS 8
#define COLS 8
#define MASTER_THREAD 0

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

void initializeMatrix(int rows, int cols, int initialValue, int ** matrix) {
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

void printMatrixMemAddress(int rows, int cols, int ** matrix) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%4p ", (void*)&matrix[i][j]);
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

void solveFirst(int rows, int cols, int iterations, struct timespec ts_sleep, int initialValue) {

    int instanceSize;
    MPI_Comm_size(MPI_COMM_WORLD, &instanceSize);
    int cpuRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);

    struct timeval timestamp_s;
    struct timeval timestamp_e;

    // initialize Matrix with initial value on root process, print it 
    int matrixSize = ROWS * COLS;
    int ** globalMatrix = NULL;

    if (cpuRank == MASTER_THREAD){
        gettimeofday(&timestamp_s, NULL);
        globalMatrix = allocateMatrix(ROWS, COLS);
        initializeMatrix(ROWS, COLS, initialValue, globalMatrix);
        printMatrix(ROWS, COLS, globalMatrix);
    }

    // Subset buffer initialization
    // get number of cells in each process
    // ex: if matrix is 64, instance/proc size 16, each process (subMatrix) manages 4 cells
    int numberOfCellsPerProcessor = (matrixSize) / instanceSize;
    int dimensionPerProcessor = numberOfCellsPerProcessor / 2;
    // int *subMatrix = (int *)malloc(sizeof(int) * numberOfCellsPerProcessor);

    int **subMatrix = allocateMatrix(numberOfCellsPerProcessor/2, numberOfCellsPerProcessor/2);

    // subtype creation...
    int globalDimensions[2] = {ROWS,COLS};
    int subDimensions[2] = {dimensionPerProcessor, dimensionPerProcessor};
    int startingPosition[2] = {0,0};
    MPI_Datatype type, subType;
    MPI_Type_create_subarray(2, globalDimensions, subDimensions, startingPosition, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, (dimensionPerProcessor)*sizeof(int), &subType);
    MPI_Type_commit(&subType);

    int *globalMatrixPointer=NULL;
    if (cpuRank == MASTER_THREAD)
        globalMatrixPointer = &(globalMatrix[0][0]);

    int scatterCount[numberOfCellsPerProcessor];
    int displacement[numberOfCellsPerProcessor];

    if (cpuRank == MASTER_THREAD){

        for (int i = 0; i < numberOfCellsPerProcessor; i++){
            scatterCount[i] = 1;
        }

        int displacementIterator = 0;

        for (int i = 0; i < dimensionPerProcessor; i++){
            for (int j = 0; j < dimensionPerProcessor; j++){
                displacement[i * dimensionPerProcessor * j] = displacementIterator;
                displacementIterator++;
            }

            displacementIterator += (numberOfCellsPerProcessor-1)*dimensionPerProcessor;
        }
    }


    // SCATTER STAGE - scatters from root proc to all process
    // int scatterStatus = MPI_Scatter(globalMatrix, 
    //                                 numberOfCellsPerProcessor, 
    //                                 MPI_INT, 
    //                                 subMatrix, 
    //                                 numberOfCellsPerProcessor, 
    //                                 MPI_INT, 
    //                                 0, 
    //                                 MPI_COMM_WORLD);

    // if (scatterStatus != MPI_SUCCESS){
    //     printf("[Error] MPI_Scatter\n");
    // }

    // MPI_Scatterv(globalMatrixPointer, 
    //             scatterCount, 
    //             displacement, 
    //             subType, 
    //             &(subMatrix[0][0]), 
    //             instanceSize,
    //             MPI_INT,
    //             0,
    //             MPI_COMM_WORLD);

    // for (int p=0; p<instanceSize; p++) {
    //     if (cpuRank == p) {
    //         printf("Local process on rank %d is:\n", cpuRank);
    //         for (int i=0; i<instanceSize; i++) {
    //             putchar('|');
    //             for (int j=0; j<instanceSize; j++) {
    //                 printf("%2d ", subMatrix[i][j]);
    //             }
    //             printf("|\n");
    //         }
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    //printf("CPU: %3d, Size: %3d, subMatSize: %3d \n", cpuRank, instanceSize, numberOfCellsPerProcessor);

    // for(int k = 1; k <= iterations; k++) {
    //     for(int j = 0; j < cols; j++) {
    //         for(int i = 0; i < rows; i++) {
    //             usleep(1000);
    //             matrix[i][j] = matrix[i][j] + (i + j) * k;
    //         }
    //     }
    // }

    // for(int k = 1; k <= iterations; k++) {
    //     for(int j = 0; j < numberOfCellsPerProcessor; j++) {
    //         for(int i = 0; i < numberOfCellsPerProcessor; i++) {
    //             usleep(1000);
    //             subMatrix[i] = subMatrix[i] + (i + j) * k;
    //         }
    //     }
    // }

    int mpiGatherResult = MPI_Gather(subMatrix, 
                                numberOfCellsPerProcessor, 
                                MPI_INT, 
                                globalMatrix, 
                                numberOfCellsPerProcessor, 
                                MPI_INT, 
                                0, 
                                MPI_COMM_WORLD);

    if (mpiGatherResult != MPI_SUCCESS){
        printf("[Error] MPI_Gather\n");
    }

    if (cpuRank == MASTER_THREAD){
        
        gettimeofday(&timestamp_e, NULL);
        printMatrix(ROWS, COLS, globalMatrix);
        printRuntime(timestamp_s, timestamp_e);
        deallocateMatrix(ROWS, globalMatrix);
    }
}

void solveSecond(int rows, int cols, int iterations, struct timespec ts_sleep, int initialValue) {
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

int main(int argc, char* argv[]) {

    if(4 != argc) {
        return EXIT_FAILURE;
    }

     // Parameter initialization
    int problemChoice = atoi(argv[1]);
    int initialValue = atoi(argv[2]);
    int iterations = atoi(argv[3]);

    struct timespec ts_sleep;
    ts_sleep.tv_sec = 0;
    ts_sleep.tv_nsec = 1000000L;

    // ---------------initialize MPI environment
    MPI_Init(&argc, &argv);

    if (problemChoice == 1){
        solveFirst(ROWS, COLS, iterations, ts_sleep, initialValue);
    }
    else if (problemChoice == 2){
        solveSecond(ROWS, COLS, iterations, ts_sleep, initialValue);
    }
    else{
        printf("[Error] Select valid problem choice\n");
        return EXIT_FAILURE;
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}