#ifndef MATRIX_HPP
#define MATRIX_HPP

double ** allocateMatrix(int rows, int cols);

// should work for both partialMatrix and regular matrices
void deallocateMatrix(int rows, double ** matrix);

void fillMatrix(int rows, int cols, double ** matrix);

// for dev/test purposes. same as fillMatrix but with a seed number vary values
void debug_fillMatrixWithSeed(int rows, int cols, float seed, double ** matrix);

// todo Filipe
bool cloneMatValuesAtoB(int rows, int cols, double matrixA, int rowsB, int colsB, double ** matrixB);

// todo Filipe
bool isMatEqual(int rowsA, int colsA, double ** matrixA, int rowsB, int colsB, double ** matrixB);

// todo Howard
    // EXAMPLE 1
    // ┌─────────────┐
    // │ 0 0 0 0 0 0 │
    // │ 0 1 2 2 1 0 │
    // │ 0 3 4 4 3 0 │
    // │ 0 3 4 4 3 0 │
    // │ 0 1 2 2 1 0 │
    // │ 0 0 0 0 0 0 │
    // └─────────────┘
    // ┌─────┐
    // │ 1 2 │
    // │ 3 4 │
    // └─────┘

    // EXAMPLE 2
    // ┌───────────────────┐
    // │ 0   0   0   0   0 │
    // │ 0   9  12   9   0 │
    // │ 0  12  16  12   0 │
    // │ 0   9  12   9   0 │
    // │ 0   0   0   0   0 │
    // └───────────────────┘
    // ┌────────┐
    // │  9  12 │
    // │ 12  16 │
    // └────────┘
double ** allocatePartialMatFromTargetMat(int * pmRows, int * pmCols, double ** partialMatrix, int tmRows, int tmCols, double targetMatrix);

// todo Filipe
bool mirrorPartialMatToTargetMat(int pmRows, int pmCols, double partialMatrix, int tmRows, int tmCols, double ** targetMatrix);
    // takes a partial matrix and mirrors it to the remaining 3 quadrants. returns true if successful.
    // must adapt to various sizes

    // example...if partialMatrix is 2x2, targetMatrix est 6x6
    // ┌─────┐
    // │ 1 2 │
    // │ 3 4 │
    // └─────┘
    // - zeroes at the borders
    // - mirroring of values in quadrant 2, 3, 4
    // ┌─────────────┐
    // │ 0 0 0 0 0 0 │
    // │ 0 1 2 2 1 0 │
    // │ 0 3 4 4 3 0 │
    // │ 0 3 4 4 3 0 │
    // │ 0 1 2 2 1 0 │
    // │ 0 0 0 0 0 0 │
    // └─────────────┘

    // example...if partialMatrix is 2x2, targetMatrix est 5x5
    // ┌────────┐
    // │  9  12 │
    // │ 12  16 │
    // └────────┘
    // ┌───────────────────┐
    // │ 0   0   0   0   0 │
    // │ 0   9  12   9   0 │
    // │ 0  12  16  12   0 │
    // │ 0   9  12   9   0 │
    // │ 0   0   0   0   0 │
    // └───────────────────┘

    // example 2 ... avec une partialMatrix de 3x5, targetMatrix 7x11
    // ┌─────────────┐
    // │  45  72  81 │
    // │  80 128 144 │
    // │ 105 168 189 │
    // │ 120 192 216 │
    // │ 125 200 225 │
    // └─────────────┘
    // - same conditions like previous example, mais notez quels+comment valeurs sont en miroir, surtout dans les cas targetMatrix impair
    // ┌──────────────────────────┐
    // │ 0   0   0   0   0   0  0 │
    // │ 0  45  72  81  72  45  0 │
    // │ 0  80 128 144 128  80  0 │
    // │ 0 105 168 189 168 105  0 │
    // │ 0 120 192 216 192 120  0 │
    // │ 0 125 200 225 200 125  0 │
    // │ 0 120 192 216 192 120  0 │
    // │ 0 105 168 189 168 105  0 │
    // │ 0  80 128 144 128  80  0 │
    // │ 0  45  72  81  72  45  0 │
    // │ 0   0   0   0   0   0  0 │
    // └──────────────────────────┘

    // example 3 ... avec une partialMatrix de 6x2, targetMatrix 13x6
    // ┌────────────────────────┐
    // │ 10  18  26  31  35  36 │
    // │ 16  30  42  51  57  58 │
    // └────────────────────────┘
    // ┌───────────────────────────────────────────────────┐
    // │ 0   0   0   0   0   0   0   0   0   0   0   0   0 │
    // │ 0  10  18  26  31  35  36  35  31  26  18  10   0 │
    // │ 0  16  30  42  51  57  58  57  51  42  30  16   0 │
    // │ 0  16  30  42  51  57  58  57  51  42  30  16   0 │
    // │ 0  10  18  26  31  35  36  35  31  26  18  10   0 │
    // │ 0   0   0   0   0   0   0   0   0   0   0   0   0 │
    // └───────────────────────────────────────────────────┘

    // example 4 ... avec une partialMatrix de 6x2, targetMatrix 14x6
    // ┌────────────────────────┐
    // │ 11  21  30  36  41  43 │
    // │ 18  34  48  59  66  70 │
    // └────────────────────────┘
    // ┌───────────────────────────────────────────────────────┐
    // │ 0   0   0   0   0   0   0   0   0   0   0   0   0   0 │
    // │ 0  11  21  30  36  41  43  43  41  36  30  21  11   0 │
    // │ 0  18  34  48  59  66  70  70  66  59  48  34  18   0 │
    // │ 0  18  34  48  59  66  70  70  66  59  48  34  18   0 │
    // │ 0  11  21  30  36  41  43  43  41  36  30  21  11   0 │
    // │ 0   0   0   0   0   0   0   0   0   0   0   0   0   0 │
    // └───────────────────────────────────────────────────────┘



#endif
