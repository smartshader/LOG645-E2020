#include <chrono>
#include <cstring>
#include <thread>

#include <mpi.h>

#include "solver.hpp"
#include "../matrix/matrix.hpp"

using std::memcpy;

using std::this_thread::sleep_for;
using std::chrono::microseconds;


void solvePar(int rows, int cols, int iterations, double td, double h, int sleep, double ** matrix) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
/*
    if(0 != rank) {
        deallocateMatrix(rows, matrix);
    }

    sleep_for(microseconds(500000));
	
*/	
	oneCellOneCPU(rows,cols, iterations, td, h, sleep, rows * cols, matrix);
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

void oneCellOneCPU(int rows, int cols, int iterations, double td, double h, int sleep, int nbCells, double ** matrix) {
	
	/*
	Faire iteration comme exterieur.
	
	Appel MPI_Scatter de copie de matrix
	Recuperer valeurs de copie de matrix (valeurs precedentes) qui sont top, left, right et bottom, puis calculer champ
	Attendre tous pour fin calcul, puis MPI_Gatter sur copie matrice
	Faire copie de nouvelle matrice.
	
	Position i = rank % total nb rows
	Position j = tank / total nb rows
	
	*/
	int subMatrix[1][1] {};
	
	int rank;	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int rankPosI = rank % rows;
	int rankPosJ = rank / rows;
	
	int rankTop = ((rankPosJ - 1) * rows) + rankPosI;
	int rankBottom = ((rankPosJ + 1) * rows) + rankPosI;
	int rankLeft = (rankPosJ * rows) + rankPosI - 1;
	int rankRight = (rankPosJ * rows) + rankPosI + 1;
	
	for(int k = 0; k < iterations; k++) {
		
		int scatterStatus = MPI_Scatter(&matrix,
										1,
										MPI_DOUBLE,
										&subMatrix,
										1,
										MPI_DOUBLE,
										0,
										MPI_COMM_WORLD);
										
		if (scatterStatus != MPI_SUCCESS)
		{
			printf("[Error] MPI_Scatter\n");
			return;
		}									
		
		
	}
	
}