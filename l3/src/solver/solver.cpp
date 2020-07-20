#include <chrono>
#include <cstring>
#include <thread>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <mpi.h>

#include "solver.hpp"
#include "../matrix/matrix.hpp"
#include "../output/output.hpp"

#define GREEN "\033[32m"
#define MAGENTA "\033[35m"
#define RED "\033[31m"
#define RESET "\033[0m"
#define MASTER_CPU 0

using std::cout;
using std::endl;
using std::fixed;
using std::flush;
using std::setprecision;
using std::setw;

using std::memcpy;
using std::chrono::microseconds;
using std::this_thread::sleep_for;

// STRATEGY
// - divide the principal matrix into 1/4
// - identify its partial rows and col.
// - calculate the total number of cells of this 1/4 matrix
// - generate a struct map that contains the total number of cells found to store its coordinates
// 		- take note that i,j coordinates should NEVER be 0 because we do not calculate borders
// - create cell partitions that are based on the total number of cells divided by total cpus
// - generate room[x][y], where x = total cpus and y = cell partitions
//		- current cpuRank would be accessed as room[cpuRank][y]
//		- the cell (of main matrix resides in room[x][cell])
//		- holds reference info of original matrix
//	- generate subMatrixes to store calculated information
// 	- generate totalMatrix which agglomerates all subMatrixes using MPI_Allgather
//	- copy totalMatrix to main matrix for next iteration
void solvePar(int rows, int cols, int iterations, double td, double h, int sleep, double **matrix)
{
	double h_area = h * h;

	// get instance data
	int instanceSize,cpuRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &cpuRank);
	MPI_Comm_size(MPI_COMM_WORLD, &instanceSize);

	// calculate partial rows and cols matrix
	int partialRow = (rows % 2 == 0) ? rows / 2 : rows / 2 + 1;
	int partialCol = (cols % 2 == 0) ? cols / 2 : cols / 2 + 1;

	// calculate the total number of tiles to process, instanciate them
	int totalCells = (partialRow - 1) * (partialCol - 1);

	// used as a reference to store our original coordinates
	struct position{
		int i, j;
	} cellMap[totalCells];

	int address = 0;
	for (int i = 1; i < partialRow; i++)
	{
		for (int j = 1; j < partialCol; j++)
		{
			if (j != cols - 1 &&
				j != 0 &&
				i != 0)
			{
				cellMap[address].i = i;
				cellMap[address].j = j;
				address++;
			}
		}
	}

	// partition is based on number of total cells and how much CPUs the instance has
	int cellPartition = (totalCells % instanceSize == 0) ? totalCells / instanceSize : totalCells / instanceSize + 1;

	// each processor is then assigned a cell partition
	int room[instanceSize][cellPartition];

	int roomID = 0;

	// set identifier to room cells that contain values, -1 if blank
	for (int cpuRanking = 0; cpuRanking < instanceSize; cpuRanking++)
	{
		for (int cell = 0; cell < cellPartition; cell++)
		{
			room[cpuRanking][cell] = (roomID < totalCells) ? roomID : -1;
			roomID++;
		}
	}

	// every subMatrix should contain [x,y,value]
	double *subMatrix = (double *)malloc(sizeof(double) * cellPartition * 3);

	for (int i = 0; i < iterations; i++)
	{

		int stride = 0;

		for (int cell = 0; cell < cellPartition; cell++)
		{
			// ignore all empty cells
			if (room[cpuRank][cell] != -1)
			{

				// get data from cellMap and matrix
				int x = cellMap[room[cpuRank][cell]].i;
				int y = cellMap[room[cpuRank][cell]].j;
				double l = matrix[x][y - 1];
				double r = matrix[x][y + 1];
				double t = matrix[x - 1][y];
				double b = matrix[x + 1][y];
				double c = matrix[x][y];

				// calculate and register coordinates to submatrix
				subMatrix[stride] = double(x);
				subMatrix[stride + 1] = double(y);
				sleep_for(microseconds(sleep));
				subMatrix[stride + 2] = (1.0 - 4.0 * td / h_area) * c + (td / h_area) * (t + b + l + r);
			}
			else
			{
				subMatrix[stride] = (double)-1;
				subMatrix[stride + 1] = (double)-1;
				subMatrix[stride + 2] = (double)-1;
			}
			stride += 3;
		}

		// generate totalMatrix to gather all subMatrix
		double *totalMatrix = (double *)malloc(sizeof(double) * stride * instanceSize);

		// gathers all subMatrixes to totalMatrix to be copied to our targetMatrix
		MPI_Allgather(subMatrix, cellPartition * 3, MPI_DOUBLE, totalMatrix, stride, MPI_DOUBLE, MPI_COMM_WORLD);
		
		// copy to the entire matrix for next iteration
		copyTotalMatrixToTargetMatrix(rows, cols, stride * instanceSize, totalMatrix, matrix);
	}

	free(subMatrix);

	if (cpuRank != MASTER_CPU)
	{
		deallocateMatrix(rows, matrix);
	}
}

// used to copy values from totalMatrix to targetMatrix after an MPI allgather call
void copyTotalMatrixToTargetMatrix(int rows, int cols, int sizeTotalMatrix, double *totalMatrix, double **matrix){
	for (int currentTotalMatrix = 0; currentTotalMatrix < sizeTotalMatrix; currentTotalMatrix += 3) {

		if ((int) totalMatrix[currentTotalMatrix] != -1) {

			int x = (int) totalMatrix[currentTotalMatrix];
			int y = (int) totalMatrix[currentTotalMatrix + 1];
			double value = totalMatrix[currentTotalMatrix + 2];

			matrix[x][y] = value;
			matrix[x][cols - 1 - y] = value;
			matrix[rows - 1 - x][y] = value;
			matrix[rows - 1 - x][cols - 1 - y] = value;
		}
	}
}

void solveSeq(int rows, int cols, int iterations, double td, double h, int sleep, double ** matrix) {
    double c, l, r, t, b;
    
    double h_square = h * h;

    double * linePrevBuffer = new double[cols];
    double * lineCurrBuffer = new double[cols];

    for(int k = 0; k < iterations; k++) {

        memcpy(linePrevBuffer, matrix[0], cols * sizeof(double));
        for(int i = 1; i < rows - 1; i++) {

            memcpy(lineCurrBuffer, matrix[i], cols * sizeof(double));
            for(int j = 1; j < cols - 1; j++) {
                c = lineCurrBuffer[j];
                t = linePrevBuffer[j];
                b = matrix[i + 1][j];
                l = lineCurrBuffer[j - 1];
                r = lineCurrBuffer[j + 1];

                sleep_for(microseconds(sleep));
                matrix[i][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }

            memcpy(linePrevBuffer, lineCurrBuffer, cols * sizeof(double));
        }
    }


}