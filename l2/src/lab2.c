#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

#include <omp.h>

#define ROWS 12
#define COLS 12
#define TIMEWAIT 50000L

int ** allocateMatrix(int rows, int cols) {
    int ** matrix = (int **) malloc(rows * sizeof(int *));

    for(int i = 0; i < rows; i++) {
        matrix[i] = (int *) malloc(cols * sizeof(int));
    }

    return matrix;
}

int * allocateVecMatrix(int length) {
    int * matrix = (int *) malloc(length * sizeof(int *));

    for(int i = 0; i < length; i++) {
        matrix[i] = malloc(sizeof(int));
    }

    return matrix;
}

void deallocateMatrix(int rows, int ** matrix) {
    for(int i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix);
}

void deallocateVecMatrix(int * matrix) {
    free(matrix);
}

void fillMatrix(int rows, int cols, int initialValue, int ** matrix) {
     for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            matrix[i][j] = initialValue;
        }
    }
}

void fillVecMatrix(int length, int initialValue, int *matrix){
    for(int i = 0; i < length; i++){
        matrix[i] = initialValue;
    }
}


void printMatrix(int rows, int cols, int ** matrix) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%6d ", matrix[i][j]);
        }

        printf("\n");
    }

    printf("\n");
}

void printLinMatrix(int rows, int cols, int * matrix) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%6d ", matrix[i * rows + j]);
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


    // filipe's version
// #pragma omp parallel
//     for (int k = 1; k <= iterations; k++)
//     {
// #pragma omp for schedule(static) nowait
//         for (int i = 0; i < rows; i++)
//         {
//             for (int j = 0; j < cols; j++)
//             {

//                 // usleep(1000);
//                 nanosleep(&ts_sleep, NULL);
//                 matrix[i][j] = matrix[i][j] + i + j;
//             }
//         }
//     }

// in prog
    int tempTotal;
#pragma omp parallel \
shared(matrix) \
private(tempTotal)
{
    #pragma omp for collapse(2)
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 1; k <= iterations; k++)
            {
                tempTotal += + i + j;
            }

            // sol 1 - this is faster than sol 2
            usleep(1000);
            #pragma omp atomic
            matrix[i][j] +=tempTotal;
            tempTotal = 0;

            // // sol 2
            // #pragma omp critical
            // {
            //     usleep(1000);
            //     matrix[i][j] +=tempTotal;
            //     tempTotal = 0;
            // }
            
        }
    }
}



    // int i, j, k;
	// #pragma omp parallel for collapse(3)\
    // shared(matrix) \
    // private(i, j, k)
    // // # pragma omp for
    // // #pragma omp for schedule(guided) nowait
	// for (k = 1; k <= iterations; k++)
	// {
	// 	// #pragma omp for schedule(static) nowait
    //     //#pragma omp for reduction ( +:matrix)
	// 	for ( j = 0; j < cols; j++)
	// 	{
	// 		for ( i = 0; i < rows; i++)
	// 		{
    //             nanosleep(&ts_sleep, NULL);
	// 			// usleep(SECONDS);
	// 			matrix[i][j] = matrix[i][j] + i + j;
	// 		}
	// 	}
	// }

//     int i, j, k, tempTotal_i, tempTotal_j, tempTotal;
// #pragma omp parallel for shared(matrix, i, j, tempTotal) private(k)
//     // #pragma omp parallel for nowait \
//     // shared(matrix, i, j, tempTotal) \
//     // private(k)
//     // # pragma omp for
//     // #pragma omp for schedule(guided) nowait
//     // #pragma omp for schedule(static) nowait
//     //#pragma omp for reduction ( +:matrix)
//     for (k = 1; k <= iterations; k++)
//     {
//         // #pragma omp for nowait
//         for (i = 0; i < rows; i++)
//         {
//             for (j = 0; j < cols; j++)
//             {
//                 // #pragma omp for reduction ( +:tempTotal)
//                 // #pragma omp for
//                 // #pragma omp for

//                 nanosleep(&ts_sleep, NULL);
//                 //usleep(SECONDS);
//                 // tempTotal += i + j;
//                 matrix[i][j] = matrix[i][j] + i + j;

//                 // usleep(SECONDS);
//                 // #pragma omp critical
//                 // {
//                 //     usleep(SECONDS);
//                 //     matrix[i][j] += tempTotal;
//                 //     tempTotal = 0;
//                 // }
//             }
//             // tempTotal = 0;
//         }
//     }
}

void solveSecond(const int rows, const int cols, const int iterations, const struct timespec ts_sleep, int ** matrix) {
	#pragma omp parallel
    for (int k = 1; k <= iterations; k++)
	{
		#pragma omp for nowait
		for (int i = 0; i < rows; i++)
		{
			for (int j = cols - 1; j >= 0; j--)
			{
				if (j == cols - 1)
				{
                    nanosleep(&ts_sleep, NULL);
					// usleep(MILLISECONDS);
					matrix[i][j] = matrix[i][j] + i;
				}
				else
				{
                    nanosleep(&ts_sleep, NULL);
					// usleep(MILLISECONDS);
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
    ts_sleep.tv_nsec = TIMEWAIT;

    int nbThreads = atoi(argv[1]);
    int problem = atoi(argv[2]);
    int initialValue = atoi(argv[3]);
    int iterations = atoi(argv[4]);

    void * solvers[2];
    solvers[0] = solveFirst;
    solvers[1] = solveSecond;

    solve = solvers[problem - 1];

    int ** matrix = allocateMatrix(ROWS, COLS);
    // int * linearMatrix = allocateVecMatrix(ROWS * COLS);
    // fillVecMatrix(ROWS*COLS,initialValue, linearMatrix);
    // printLinMatrix(ROWS,COLS,linearMatrix);


    // _______________________ Sequential
    struct timeval timestamp_s_seq;
    struct timeval timestamp_e_seq;

    omp_set_num_threads(1);
    fillMatrix(ROWS, COLS, initialValue, matrix);

    printf("[Sequential : Prob = #%d, procs = %d, threads = 1, initValue = %d, iterations = %d ]\n", problem, omp_get_num_procs(), initialValue, iterations);

    gettimeofday(&timestamp_s_seq, NULL);
    //solve(ROWS, COLS, iterations, ts_sleep, matrix);
    gettimeofday(&timestamp_e_seq, NULL);
    
    printMatrix(ROWS, COLS, matrix);

    // _______________________ Parallel
    struct timeval timestamp_s_par;
    struct timeval timestamp_e_par;

    omp_set_num_threads(nbThreads);
    fillMatrix(ROWS, COLS, initialValue, matrix);

    printf("[Parallel : Prob = #%d, procs = %d, threads = %d, initValue = %d, iterations = %d ]\n", problem, omp_get_num_procs(), nbThreads, initialValue, iterations);

    gettimeofday(&timestamp_s_par, NULL);
    solve(ROWS, COLS, iterations, ts_sleep, matrix);
    gettimeofday(&timestamp_e_par, NULL);
    
    printMatrix(ROWS, COLS, matrix);

    // _______________________Statistics
    printStatistics(nbThreads, timestamp_s_seq, timestamp_e_seq, timestamp_s_par, timestamp_e_par);
    deallocateMatrix(ROWS, matrix);

    return EXIT_SUCCESS;
}
