#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <string.h>

#include "solver.h"

int max(int a, int b);
int min(int a, int b);

void solveFirst(const int rows, const int cols, const int iterations, const struct timespec ts_sleep, int ** matrix) {

}

void solveSecond(const int rows, const int cols, const int iterations, const struct timespec ts_sleep, int ** matrix) {

}

int max(int a, int b) {
    return a >= b ? a : b;
}

int min(int a, int b) {
    return a <= b ? a : b;
}
