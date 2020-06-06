#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

#define ROWS 8
#define COLS 8
#define MASTER_THREAD 0
#define LEN_PROCESSMATRIX 4
#define LEN_MATRIX 8

struct position{
    int i_position;
    int j_position;
}cpuCoordinates[LEN_PROCESSMATRIX*LEN_PROCESSMATRIX];

int ** allocateMatrix(int rows, int cols) {

    // necessary to allocate contiguously
    int *temp = (int *)malloc(rows*cols*sizeof(int));
    int ** matrix = (int **) malloc(rows * sizeof(int *));
    for(int i = 0; i < rows; i++)
        matrix[i] = &(temp[cols*i]);

    return matrix;
}

void deallocateMatrix(int rows, int ** matrix) {
    free(matrix[0]);
    free(matrix);
}

void initializeCPUGrid(){
    int cellId = 0;
    for (int i = 0; i < LEN_PROCESSMATRIX; ++i){
        for (int j = 0; j < LEN_PROCESSMATRIX; ++j){
            cpuCoordinates[cellId].i_position = i;
            cpuCoordinates[cellId].j_position = j;
            ++cellId;
        }
    }
}

void initializeMatrix(int rows, int cols, int initialValue, int ** matrix) {
     for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) 
            matrix[i][j] = initialValue;
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

void debugPrintSubMatrix(int cpuRank, int instanceSize, int **subMatrix){

    for (int p=0; p<instanceSize; p++) {

        if (cpuRank == p) {

            printf("Local process on rank %d [%2d,%2d] is: \n", cpuRank, cpuCoordinates[cpuRank].i_position, cpuCoordinates[cpuRank].j_position);
            for (int i=0; i < ROWS/LEN_PROCESSMATRIX; i++) {
                putchar('|');
                for (int j=0; j< ROWS/LEN_PROCESSMATRIX; j++) {
                    printf("%6d ", subMatrix[i][j]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void solveFirst(int rows, int cols, int iterations, struct timespec ts_sleep, int initialValue) {

    int instanceSize;
    MPI_Comm_size(MPI_COMM_WORLD, &instanceSize);
    int cpuRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);

    struct timeval timestamp_s;
    struct timeval timestamp_e;

    // initialize Matrix with initial value on root process, print it 
    int ** globalMatrix = NULL;
    int ** subMatrix = NULL;

    if (cpuRank == MASTER_THREAD){
        gettimeofday(&timestamp_s, NULL);
        globalMatrix = allocateMatrix(ROWS, COLS);
        initializeMatrix(ROWS, COLS, initialValue, globalMatrix);
    }

    // subMatrix buffer initialization
    // get number of cells in each process
    // ex: if matrix is 64, instance/proc size 16, each process (subMatrix) manages 4 cells
    int lengthOfGlobalMatrix = ROWS;
    int lengthOfProcessingGrid = LEN_PROCESSMATRIX;
    int lengthOfProcessor = lengthOfGlobalMatrix/lengthOfProcessingGrid;

    // subType creation
    int globalDimensions[2] = {ROWS,COLS};
    int subDimensions[2] = {lengthOfProcessor, lengthOfProcessor};
    int startingPosition[2] = {0,0};
    MPI_Datatype type, subType;
    MPI_Type_create_subarray(2, globalDimensions, subDimensions, startingPosition, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, (lengthOfProcessor)*sizeof(int), &subType);

    MPI_Type_commit(&subType);

    // variables required for MPI_Scatterv operations
    // pointer is used to set to globalMatrix 
    int *globalMatrixPointer = NULL;
    // counts relative to number of CPUs
    int sendCounts[LEN_PROCESSMATRIX*LEN_PROCESSMATRIX];
    // displacement between CPUs
    int displacement[LEN_PROCESSMATRIX*LEN_PROCESSMATRIX];

    if (cpuRank == MASTER_THREAD){
        
        globalMatrixPointer = &(globalMatrix[0][0]);

        for (int i = 0; i < LEN_PROCESSMATRIX*LEN_PROCESSMATRIX; i++)
            sendCounts[i] = 1;
        
        // measures displacement between CPUs
        int offsetDisp = 0;

        for (int i = 0; i < lengthOfProcessingGrid; i++){
            for (int j = 0; j < lengthOfProcessingGrid; j++){
                displacement[i * lengthOfProcessingGrid + j] = offsetDisp;
                offsetDisp++;
            }
            offsetDisp += ((ROWS/LEN_PROCESSMATRIX)-1)*LEN_PROCESSMATRIX;
        }
    }

    subMatrix = allocateMatrix(lengthOfProcessor, lengthOfProcessor);

    // SCATTERV STAGE - creates subMatrixes from globalMatrix
    int scatterStatus = MPI_Scatterv(globalMatrixPointer, 
                                        sendCounts, 
                                        displacement, 
                                        subType, 
                                        &(subMatrix[0][0]), 
                                        ROWS*COLS/(LEN_PROCESSMATRIX*LEN_PROCESSMATRIX),
                                        MPI_INT,
                                        0,
                                        MPI_COMM_WORLD);

    if (scatterStatus != MPI_SUCCESS){
        printf("[Error] MPI_Scatter\n");
        return;
    }

    // subMatrix calculations

    // extract i,j of cpuCoordinates
    int i_origin = cpuCoordinates[cpuRank].i_position*2;
    int j_origin = cpuCoordinates[cpuRank].j_position*2;

    for(int k = 1; k <= iterations; k++) {
        for(int j = 0; j < ROWS/LEN_PROCESSMATRIX; j++) {
            for(int i = 0; i < ROWS/LEN_PROCESSMATRIX; i++) {
                usleep(1000);
                subMatrix[i][j] = subMatrix[i][j] + (i_origin + i + j_origin + j)*k;
            }
        }
    }

    // gathers all subMatrixes back to globalMatrix
    MPI_Gatherv(&(subMatrix[0][0]), 
                ROWS*COLS/(LEN_PROCESSMATRIX*LEN_PROCESSMATRIX),  
                MPI_INT,
                globalMatrixPointer, 
                sendCounts, 
                displacement, 
                subType,
                0, 
                MPI_COMM_WORLD);
    
    MPI_Type_free(&subType);

    // Display results in final thread
    if (cpuRank == MASTER_THREAD){
        gettimeofday(&timestamp_e, NULL);
        printMatrix(ROWS, COLS, globalMatrix);
        printRuntime(timestamp_s, timestamp_e);
        deallocateMatrix(ROWS, globalMatrix);
        deallocateMatrix(lengthOfProcessor, subMatrix);
    }
}

void solveSecond(int rows, int cols, int iterations, struct timespec ts_sleep, int initialValue) {

    int instanceSize;
    MPI_Comm_size(MPI_COMM_WORLD, &instanceSize);
    int cpuRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);

    struct timeval timestamp_s;
    struct timeval timestamp_e;

    // initialize Matrix with initial value on root process, print it 
    int ** globalMatrix = NULL;
    int ** subMatrix = NULL;
    int ** receivedSubMatrix = NULL;
    int * globalMatrixPointer = NULL;

    if (cpuRank == MASTER_THREAD){
        gettimeofday(&timestamp_s, NULL);
        globalMatrix = allocateMatrix(ROWS, COLS);
        initializeMatrix(ROWS, COLS, initialValue, globalMatrix);
    }

    // subMatrix buffer initialization
    // get number of cells in each process
    // ex: if matrix is 64, instance/proc size 16, each process (subMatrix) manages 4 cells
    int lengthOfGlobalMatrix = ROWS;
    int lengthOfProcessingGrid = LEN_PROCESSMATRIX;
    int lengthOfProcessor = lengthOfGlobalMatrix/lengthOfProcessingGrid;

    // subType creation
    int globalDimensions[2] = {ROWS,COLS};
    int subDimensions[2] = {lengthOfProcessor, lengthOfProcessor};
    int startingPosition[2] = {0,0};
    MPI_Datatype type, subType;
    MPI_Type_create_subarray(2, globalDimensions, subDimensions, startingPosition, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, (lengthOfProcessor)*sizeof(int), &subType);
    MPI_Type_commit(&subType);

    int sendCounts[LEN_PROCESSMATRIX*LEN_PROCESSMATRIX];
    int displacement[LEN_PROCESSMATRIX*LEN_PROCESSMATRIX];

    if (cpuRank == MASTER_THREAD){
        globalMatrixPointer = &(globalMatrix[0][0]);

        for (int i = 0; i < LEN_PROCESSMATRIX*LEN_PROCESSMATRIX; i++){
            sendCounts[i] = 1;
        }

        // submatrix dispalcement
        int offsetDisp = 0;

        for (int i = 0; i < lengthOfProcessingGrid; i++){
            for (int j = 0; j < lengthOfProcessingGrid; j++){
                displacement[i * lengthOfProcessingGrid + j] = offsetDisp;
                offsetDisp++;
            }
            offsetDisp += ((ROWS/LEN_PROCESSMATRIX)-1)*LEN_PROCESSMATRIX;
        }
    }

    subMatrix = allocateMatrix(lengthOfProcessor, lengthOfProcessor);

    int scatterStatus = MPI_Scatterv(globalMatrixPointer, 
                                        sendCounts, 
                                        displacement, 
                                        subType, 
                                        &(subMatrix[0][0]), 
                                        ROWS*COLS/(LEN_PROCESSMATRIX*LEN_PROCESSMATRIX),
                                        MPI_INT,
                                        0,
                                        MPI_COMM_WORLD);

    if (scatterStatus != MPI_SUCCESS){
        printf("[Error] MPI_Scatter\n");
        return;
    }

    // extract i coordinates, relative to cpuRank
    int i_origin = cpuCoordinates[cpuRank].i_position*2;

    receivedSubMatrix = allocateMatrix(lengthOfProcessor, lengthOfProcessor);

    for(int k = 1; k <= iterations; k++) {

        // initialize the first column cpuCoordinates
        if (cpuCoordinates[cpuRank].j_position == 0){

            for(int i = 0; i < ROWS/LEN_PROCESSMATRIX; i++) {
                usleep(1000);
                subMatrix[i][0] = subMatrix[i][0] + ((i_origin + i) * k);
            }

            for(int j = 1; j < ROWS/LEN_PROCESSMATRIX; j++) {
                for(int i = 0; i < ROWS/LEN_PROCESSMATRIX; i++) {
                    usleep(1000);
                    subMatrix[i][j] = subMatrix[i][j] + subMatrix[i][j - 1] * k;
                }
            }

            // Send the processed subMatrix to the next CPU
            MPI_Send(&(subMatrix[0][0]), lengthOfProcessor*lengthOfProcessor, MPI_INT, cpuRank+1, 0, MPI_COMM_WORLD);

        }else{

            // Receive a processed subMatrix from the previous CPU
            MPI_Recv(&(receivedSubMatrix[0][0]), lengthOfProcessor*lengthOfProcessor, MPI_INT, cpuRank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(int i = 0; i < ROWS/LEN_PROCESSMATRIX; i++) {
                usleep(1000);
                subMatrix[i][0] = subMatrix[i][0] + (receivedSubMatrix[i][1] * k);
            }

            for(int j = 1; j < ROWS/LEN_PROCESSMATRIX; j++) {
                for(int i = 0; i < ROWS/LEN_PROCESSMATRIX; i++) {
                    usleep(1000);
                    subMatrix[i][j] = subMatrix[i][j] + subMatrix[i][j - 1] * k;
                }
            }

            // Send the processed subMatrix to the next CPU
            if (cpuRank < instanceSize - 1){
                MPI_Send(&(subMatrix[0][0]), lengthOfProcessor*lengthOfProcessor, MPI_INT, cpuRank+1, 0, MPI_COMM_WORLD);
            }
        }
    }

    // gathers all subMatrixes back to globalMatrix
    MPI_Gatherv(&(subMatrix[0][0]), 
                ROWS*COLS/(LEN_PROCESSMATRIX*LEN_PROCESSMATRIX),  
                MPI_INT,
                globalMatrixPointer, 
                sendCounts, 
                displacement, 
                subType,
                0, 
                MPI_COMM_WORLD);
    
    MPI_Type_free(&subType);

    // Display results in final thread
    if (cpuRank == MASTER_THREAD){
        gettimeofday(&timestamp_e, NULL);
        printMatrix(ROWS, COLS, globalMatrix);
        printRuntime(timestamp_s, timestamp_e);
        deallocateMatrix(ROWS, globalMatrix);
        deallocateMatrix(lengthOfProcessor, subMatrix);
        deallocateMatrix(lengthOfProcessor, receivedSubMatrix);
    }
}

int main(int argc, char* argv[]) {

    if(4 != argc) {
        return EXIT_FAILURE;
    }

    int problemChoice = atoi(argv[1]);
    int initialValue = atoi(argv[2]);
    int iterations = atoi(argv[3]);

    struct timespec ts_sleep;
    ts_sleep.tv_sec = 0;
    ts_sleep.tv_nsec = 1000000L;

    initializeCPUGrid();

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