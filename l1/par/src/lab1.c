#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

#define ROWS 8
#define COLS 8
#define MASTER_CPU 0
#define LEN_PROCESSMATRIX 4

// used to store grid coordinates for a 4x4 CPU matrix
struct position
{
    int i_position;
    int j_position;
} cpuCoordinates[LEN_PROCESSMATRIX * LEN_PROCESSMATRIX];

int **allocateMatrix(int rows, int cols)
{

    // necessary to allocate contiguously
    int *temp = (int *)malloc(rows * cols * sizeof(int));
    int **matrix = (int **)malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++)
        matrix[i] = &(temp[cols * i]);

    return matrix;
}

void deallocateMatrix(int rows, int **matrix)
{
    free(matrix[0]);
    free(matrix);
}

// initializes coordinates to a CPU matrix
void initializeCPUGrid()
{
    int cellId = 0;
    for (int i = 0; i < LEN_PROCESSMATRIX; ++i)
    {
        for (int j = 0; j < LEN_PROCESSMATRIX; ++j)
        {
            cpuCoordinates[cellId].i_position = i;
            cpuCoordinates[cellId].j_position = j;
            ++cellId;
        }
    }
}

// initializes our globalMatrix with appropriate initialValue
void initializeMatrix(int rows, int cols, int initialValue, int **matrix)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            matrix[i][j] = initialValue;
    }
}

void printMatrix(int rows, int cols, int **matrix)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%12d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printRuntime(struct timeval tvs, struct timeval tve)
{
    long start = tvs.tv_sec * 1000000 + tvs.tv_usec;
    long end = tve.tv_sec * 1000000 + tve.tv_usec;
    long delta = end - start;
    printf("Runtime: %.6f seconds\n", delta / 1000000.0);
}

// Prints a 2x2 subMatrix for debugging purposes
void debugPrintSubMatrix(int cpuRank, int instanceSize, int **subMatrix)
{

    for (int p = 0; p < instanceSize; p++)
    {

        if (cpuRank == p)
        {

            printf("Local process on rank %d [%2d,%2d] is: \n", cpuRank, cpuCoordinates[cpuRank].i_position, cpuCoordinates[cpuRank].j_position);
            for (int i = 0; i < ROWS / LEN_PROCESSMATRIX; i++)
            {
                putchar('|');
                for (int j = 0; j < ROWS / LEN_PROCESSMATRIX; j++)
                {
                    printf("%6d ", subMatrix[i][j]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// Creates and returns a new MPI_Datatype based on our globalMatrix and subMatrix parameters
MPI_Datatype createMatrixDataType(int lenSubMatrix)
{

    // define global/sub matrix dimensions
    int globalMatrixDim[2] = {ROWS, COLS};
    int subMatrixDim[2] = {lenSubMatrix, lenSubMatrix};
    // defining starting coordinates for each subMatrix
    int startingPosition[2] = {0, 0};

    MPI_Datatype type, subType;

    // create the MPI_Datatype based on parameters defined relative to our subMatrix and globalMatrix dimensions
    MPI_Type_create_subarray(2, globalMatrixDim, subMatrixDim, startingPosition, MPI_ORDER_C, MPI_INT, &type);
    // resizes our new subtype with different extent
    MPI_Type_create_resized(type, 0, (lenSubMatrix) * sizeof(int), &subType);
    // clear the original type since it's no longer needed
    MPI_Type_free(&type);
    // commit our new subtype to MPI environment
    MPI_Type_commit(&subType);

    return subType;
}

void solveFirst(int rows, int cols, int iterations, struct timespec ts_sleep, int initialValue)
{
    // get the number of processes in MPI world
    int instanceSize;
    MPI_Comm_size(MPI_COMM_WORLD, &instanceSize);
    // get the current rank of process
    int cpuRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);

    // used to store start/end times
    struct timeval timestamp_s;
    struct timeval timestamp_e;

    // a globalMatrix is our original 8x8 2D matrix
    int **globalMatrix = NULL;
    // a subMatrix is a 2x2 matrix, derived from the globalMatrix
    int **subMatrix = NULL;
    // globalMatrixRefPtr is used to store a reference to globalMatrix to when using Scatterv/Gatherv
    int *globalMatrixRefPtr = NULL;

    int lenGlobalMatrix = ROWS;
    int lenCPUGrid = LEN_PROCESSMATRIX;
    int lenSubMatrix = lenGlobalMatrix / lenCPUGrid;

    // Allocate and initialize globalMatrix when process is 0
    if (cpuRank == MASTER_CPU)
    {
        gettimeofday(&timestamp_s, NULL);
        globalMatrix = allocateMatrix(ROWS, COLS);
        initializeMatrix(ROWS, COLS, initialValue, globalMatrix);
    }

    MPI_Datatype subType = createMatrixDataType(lenSubMatrix);

    // MPI_Scatterv SETUP

    // counts relative to number of CPUs
    int sendCounts[lenCPUGrid * lenCPUGrid];
    // displacement between CPUs
    int displacement[lenCPUGrid * lenCPUGrid];

    if (cpuRank == MASTER_CPU)
    {

        globalMatrixRefPtr = &(globalMatrix[0][0]);

        for (int i = 0; i < lenCPUGrid * lenCPUGrid; i++)
            sendCounts[i] = 1;

        // measures displacement between CPUs
        int offset = 0;

        for (int i = 0; i < lenCPUGrid; i++)
        {
            for (int j = 0; j < lenCPUGrid; j++)
            {
                displacement[i * lenCPUGrid + j] = offset;
                offset++;
            }
            offset += ((ROWS / lenCPUGrid) - 1) * lenCPUGrid;
        }
    }


    subMatrix = allocateMatrix(lenSubMatrix, lenSubMatrix);

    // SCATTERV STAGE - generates subMatrixes from globalMatrix
    int scatterStatus = MPI_Scatterv(globalMatrixRefPtr,
                                     sendCounts,
                                     displacement,
                                     subType,
                                     &(subMatrix[0][0]),
                                     ROWS * COLS / (lenCPUGrid * lenCPUGrid),
                                     MPI_INT,
                                     0,
                                     MPI_COMM_WORLD);

    if (scatterStatus != MPI_SUCCESS)
    {
        printf("[Error] MPI_Scatter\n");
        return;
    }

    // PROBLEM #1 CALCULATIONS

    // Each process contains a 2x2 subMatrix that is calculated.
    // Instead of looping through a 8x8 matrix, we have 16 processes that
    // loop through a 2x2 subMatrix.

    // extract i,j of cpuCoordinates
    // after extraction, multiply by 2 as it references the globalMatrix
    // example, a cpuCoordinates of 3 is equivalent to global coordinates 6
    int i_origin = cpuCoordinates[cpuRank].i_position * 2;
    int j_origin = cpuCoordinates[cpuRank].j_position * 2;

    // Calculations
    for (int k = 1; k <= iterations; k++)
    {
        for (int j = 0; j < ROWS / lenCPUGrid; j++)
        {
            for (int i = 0; i < ROWS / lenCPUGrid; i++)
            {
                usleep(1000);
                subMatrix[i][j] = subMatrix[i][j] + (i_origin + i + j_origin + j) * k;
            }
        }
    }

    // gathers all subMatrixes back to globalMatrix
    MPI_Gatherv(&(subMatrix[0][0]),
                ROWS * COLS / (lenCPUGrid * lenCPUGrid),
                MPI_INT,
                globalMatrixRefPtr,
                sendCounts,
                displacement,
                subType,
                0,
                MPI_COMM_WORLD);

    // release the subType as we no longer need it
    MPI_Type_free(&subType);

    // Display results in final thread
    if (cpuRank == MASTER_CPU)
    {
        gettimeofday(&timestamp_e, NULL);
        printMatrix(ROWS, COLS, globalMatrix);
        printRuntime(timestamp_s, timestamp_e);
        deallocateMatrix(ROWS, globalMatrix);
        deallocateMatrix(lenSubMatrix, subMatrix);
    }
}

void solveSecond(int rows, int cols, int iterations, struct timespec ts_sleep, int initialValue)
{
     // get the number of processes in MPI world
    int instanceSize;
    MPI_Comm_size(MPI_COMM_WORLD, &instanceSize);
    // get the current rank of process
    int cpuRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);

    struct timeval timestamp_s;
    struct timeval timestamp_e;

    // globalMatrix is our main 8x8 matrix
    int **globalMatrix = NULL;
    // subMatrix is our 2x2 matrix that derives from globalMatrix
    int **subMatrix = NULL;
    // globalMatrixRefPtr is used to store a reference to globalMatrix to when using Scatterv/Gatherv
    int *globalMatrixRefPtr = NULL;

    // receivedSubMatrix is a subMatrix buffer used during our send/recv process
    int **receivedSubMatrix = NULL;

    int lenGlobalMatrix = ROWS;
    int lenCPUGrid = LEN_PROCESSMATRIX;
    int lenSubMatrix = lenGlobalMatrix / lenCPUGrid;

    // Allocate and initialize globalMatrix when process is 0
    if (cpuRank == MASTER_CPU)
    {
        gettimeofday(&timestamp_s, NULL);
        globalMatrix = allocateMatrix(ROWS, COLS);
        initializeMatrix(ROWS, COLS, initialValue, globalMatrix);
    }

    // // subType creation
    // int globalMatrixDim[2] = {ROWS, COLS};
    // int subMatrixDim[2] = {totCellsPerCPU, totCellsPerCPU};
    // int startingPosition[2] = {0, 0};
    // MPI_Datatype type, subType;
    // MPI_Type_create_subarray(2, globalMatrixDim, subMatrixDim, startingPosition, MPI_ORDER_C, MPI_INT, &type);
    // MPI_Type_create_resized(type, 0, (totCellsPerCPU) * sizeof(int), &subType);
    // MPI_Type_commit(&subType);

    MPI_Datatype subType = createMatrixDataType(lenSubMatrix);

    // counts relative to number of CPUs
    int sendCounts[lenCPUGrid * lenCPUGrid];
    // displacement between CPUs
    int displacement[lenCPUGrid * lenCPUGrid];

    if (cpuRank == MASTER_CPU)
    {
        globalMatrixRefPtr = &(globalMatrix[0][0]);

        for (int i = 0; i < lenCPUGrid * lenCPUGrid; i++)
        {
            sendCounts[i] = 1;
        }

        // measures displacement between CPUs
        int offset = 0;

        for (int i = 0; i < lenCPUGrid; i++)
        {
            for (int j = 0; j < lenCPUGrid; j++)
            {
                displacement[i * lenCPUGrid + j] = offset;
                offset++;
            }
            offset += ((ROWS / lenCPUGrid) - 1) * lenCPUGrid;
        }
    }

    subMatrix = allocateMatrix(lenSubMatrix, lenSubMatrix);

    int scatterStatus = MPI_Scatterv(globalMatrixRefPtr,
                                     sendCounts,
                                     displacement,
                                     subType,
                                     &(subMatrix[0][0]),
                                     ROWS * COLS / (lenCPUGrid * lenCPUGrid),
                                     MPI_INT,
                                     0,
                                     MPI_COMM_WORLD);

    if (scatterStatus != MPI_SUCCESS)
    {
        printf("[Error] MPI_Scatter\n");
        return;
    }

    // extract i coordinates, relative to cpuRank
    int i_origin = cpuCoordinates[cpuRank].i_position * 2;

    receivedSubMatrix = allocateMatrix(lenSubMatrix, lenSubMatrix);

    for (int k = 1; k <= iterations; k++)
    {

        // initialize the first column cpuCoordinates
        if (cpuCoordinates[cpuRank].j_position == 0)
        {

            for (int i = 0; i < ROWS / lenCPUGrid; i++)
            {
                usleep(1000);
                subMatrix[i][0] = subMatrix[i][0] + ((i_origin + i) * k);
            }

            for (int j = 1; j < ROWS / lenCPUGrid; j++)
            {
                for (int i = 0; i < ROWS / lenCPUGrid; i++)
                {
                    usleep(1000);
                    subMatrix[i][j] = subMatrix[i][j] + subMatrix[i][j - 1] * k;
                }
            }

            // Send the processed subMatrix to the next CPU
            MPI_Send(&(subMatrix[0][0]), lenSubMatrix * lenSubMatrix, MPI_INT, cpuRank + 1, 0, MPI_COMM_WORLD);
        }
        else
        {

            // Receive a processed subMatrix from the previous CPU
            MPI_Recv(&(receivedSubMatrix[0][0]), lenSubMatrix * lenSubMatrix, MPI_INT, cpuRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < ROWS / lenCPUGrid; i++)
            {
                usleep(1000);
                subMatrix[i][0] = subMatrix[i][0] + (receivedSubMatrix[i][1] * k);
            }

            for (int j = 1; j < ROWS / lenCPUGrid; j++)
            {
                for (int i = 0; i < ROWS / lenCPUGrid; i++)
                {
                    usleep(1000);
                    subMatrix[i][j] = subMatrix[i][j] + subMatrix[i][j - 1] * k;
                }
            }

            // Send the processed subMatrix to the next CPU
            if (cpuRank < instanceSize - 1)
            {
                MPI_Send(&(subMatrix[0][0]), lenSubMatrix * lenSubMatrix, MPI_INT, cpuRank + 1, 0, MPI_COMM_WORLD);
            }
        }
    }

    // gathers all subMatrixes back to globalMatrix
    MPI_Gatherv(&(subMatrix[0][0]),
                ROWS * COLS / (lenCPUGrid * lenCPUGrid),
                MPI_INT,
                globalMatrixRefPtr,
                sendCounts,
                displacement,
                subType,
                0,
                MPI_COMM_WORLD);

    MPI_Type_free(&subType);

    // Display results in final thread
    if (cpuRank == MASTER_CPU)
    {
        gettimeofday(&timestamp_e, NULL);
        printMatrix(ROWS, COLS, globalMatrix);
        printRuntime(timestamp_s, timestamp_e);
        deallocateMatrix(ROWS, globalMatrix);
        deallocateMatrix(lenSubMatrix, subMatrix);
        deallocateMatrix(lenSubMatrix, receivedSubMatrix);
    }
}

int main(int argc, char *argv[])
{

    if (4 != argc)
    {
        return EXIT_FAILURE;
    }

    int problemChoice = atoi(argv[1]);
    int initialValue = atoi(argv[2]);
    int iterations = atoi(argv[3]);

    struct timespec ts_sleep;
    ts_sleep.tv_sec = 0;
    ts_sleep.tv_nsec = 1000000L;

    // initializes a 4x4 CPU grid with identifying coordinates
    initializeCPUGrid();

    // initializes MPI space
    MPI_Init(&argc, &argv);

    if (problemChoice == 1)
    {
        solveFirst(ROWS, COLS, iterations, ts_sleep, initialValue);
    }
    else if (problemChoice == 2)
    {
        solveSecond(ROWS, COLS, iterations, ts_sleep, initialValue);
    }
    else
    {
        printf("[Error] Select valid problem choice\n");
        return EXIT_FAILURE;
    }

    // terminates MPI execution environment
    MPI_Finalize();
    return EXIT_SUCCESS;
}