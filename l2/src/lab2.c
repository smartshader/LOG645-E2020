#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#include <omp.h>

#define ROWS 12
#define COLS 12

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

void printStatistics(int nbThreads, struct timeval tvs_seq, struct timeval tve_seq, struct timeval tvs_par, struct timeval tve_par) {
    long start = tvs_seq.tv_sec * 1000000 + tvs_seq.tv_usec;
    long end = tve_seq.tv_sec * 1000000 + tve_seq.tv_usec;
    long delta_seq = end - start;

    start = tvs_par.tv_sec * 1000000 + tvs_par.tv_usec;
    end = tve_par.tv_sec * 1000000 + tve_par.tv_usec;
    long delta_par = end - start;

    double acceleration = 1.0 * delta_seq / delta_par;
    double efficiency = acceleration / nbThreads;

    printf("Runtime sequential: %.6f seconds\n", delta_seq / 1000000.0);
    printf("Runtime parallel: %.6f seconds\n", delta_par / 1000000.0);
    printf("Acceleration: %3.6f\n", acceleration);
    printf("Efficiency: %3.6f\n", efficiency);
}

int max(int a, int b);
int min(int a, int b);

void solveFirst(const int rows, const int cols, const int iterations, const struct timespec ts_sleep, int ** matrix) {
	#pragma omp parallel for
    for (int k = 1; k <= iterations; k++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int i = 0; i < rows; i++)
            {
                usleep(1000);
                matrix[i][j] = matrix[i][j] + i + j;
            }
        }
    }	
}

void solveSecond(const int rows, const int cols, const int iterations, const struct timespec ts_sleep, int ** matrix) {
	#pragma omp parallel for
    for (int k = 1; k <= iterations; k++)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = cols - 1; j >= 0; j--)
			{
				if (j == cols - 1)
				{
					matrix[i][j] = matrix[i][j] + i;
				}
				else
				{
					matrix[i][j] = matrix[i][j] + matrix[i][j+1];
				}
			}
		}
	}
}

int max(int a, int b) {
    return a >= b ? a : b;
}

int min(int a, int b) {
    return a <= b ? a : b;
}

void (* solve)(int rows, int cols, int iterations, struct timespec ts_sleep, int ** matrix) = solveFirst;

int main(int argc, char* argv[]) {
    if(5 != argc) {
        return EXIT_FAILURE;
    }

    struct timespec ts_sleep;
    ts_sleep.tv_sec = 0;
    ts_sleep.tv_nsec = 1000L;

    int nbThreads = atoi(argv[1]);
    int problem = atoi(argv[2]);
    int initialValue = atoi(argv[3]);
    int iterations = atoi(argv[4]);

    void * solvers[2];
    solvers[0] = solveFirst;
    solvers[1] = solveSecond;

    solve = solvers[problem - 1];

    int ** matrix = allocateMatrix(ROWS, COLS);

    // Sequential
    struct timeval timestamp_s_seq;
    struct timeval timestamp_e_seq;

    omp_set_num_threads(1);
    fillMatrix(ROWS, COLS, initialValue, matrix);

    gettimeofday(&timestamp_s_seq, NULL);
    solve(ROWS, COLS, iterations, ts_sleep, matrix);
    gettimeofday(&timestamp_e_seq, NULL);
    
    printMatrix(ROWS, COLS, matrix);

    // Parallel
    struct timeval timestamp_s_par;
    struct timeval timestamp_e_par;

    omp_set_num_threads(nbThreads);
    fillMatrix(ROWS, COLS, initialValue, matrix);

    gettimeofday(&timestamp_s_par, NULL);
    solve(ROWS, COLS, iterations, ts_sleep, matrix);
    gettimeofday(&timestamp_e_par, NULL);
    
    printMatrix(ROWS, COLS, matrix);

    // Statistics
    printStatistics(nbThreads, timestamp_s_seq, timestamp_e_seq, timestamp_s_par, timestamp_e_par);
    deallocateMatrix(ROWS, matrix);

    return EXIT_SUCCESS;
}
