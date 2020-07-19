#ifndef OUTPUT_HPP
#define OUTPUT_HPP

void printMatrix(int rows, int cols, double ** matrix);
void printStatistics(int threads, long runtime_seq, long runtime_par);

// for dev purposes. truncated display of results for measurements
void debug_printStatistics(int threads, long runtime_seq, long runtime_par, int rows, int cols, double ** matrixA, double ** matrixB);

#endif
