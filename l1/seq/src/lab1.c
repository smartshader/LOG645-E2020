#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>

#define ROWS 8
#define COLS 8

int **allocateMatrix(int rows, int cols)
{
    int **matrix = (int **)malloc(rows * sizeof(int *));

    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (int *)malloc(cols * sizeof(int));
    }

    return matrix;
}

void deallocateMatrix(int rows, int **matrix)
{
    for (int i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }

    free(matrix);
}

void initializeMatrix(int rows, int cols, int initialValue, int **matrix)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i][j] = initialValue;
        }
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
    //printf(" %.6f\n", delta / 1000000.0);
}

void solveFirst(int rows, int cols, int iterations, int **matrix)
{
    for (int k = 1; k <= iterations; k++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                usleep(1000);
                matrix[i][j] = matrix[i][j] + (i + j) * k;
            }
        }
    }
}

void solveSecond(int rows, int cols, int iterations, int **matrix)
{
    for (int k = 1; k <= iterations; k++)
    {
        for (int i = 0; i < rows; i++)
        {
            usleep(1000);
            matrix[i][0] = matrix[i][0] + (i * k);
        }

        for (int j = 1; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                usleep(1000);
                matrix[i][j] = matrix[i][j] + matrix[i][j - 1] * k;
            }
        }
    }
}

void (*solve)(int rows, int cols, int iterations, int **matrix) = solveFirst;

int main(int argc, char *argv[])
{
    if (4 != argc)
    {
        return EXIT_FAILURE;
    }

    struct timeval timestamp_s;
    struct timeval timestamp_e;

    int problem = atoi(argv[1]);
    int initialValue = atoi(argv[2]);
    int iterations = atoi(argv[3]);

    void *solvers[2];
    solvers[0] = solveFirst;
    solvers[1] = solveSecond;

    // solves Problem #1 or #2 based on arguments received
    solve = solvers[problem - 1];

    int **matrix = allocateMatrix(ROWS, COLS);
    initializeMatrix(ROWS, COLS, initialValue, matrix);

    gettimeofday(&timestamp_s, NULL);
    solve(ROWS, COLS, iterations, matrix);
    gettimeofday(&timestamp_e, NULL);

    printMatrix(ROWS, COLS, matrix);
    printRuntime(timestamp_s, timestamp_e);
    deallocateMatrix(ROWS, matrix);

    return EXIT_SUCCESS;
}
