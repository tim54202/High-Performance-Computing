#include <mpi.h>

#include <stdio.h>

#include <stdlib.h>

#include <time.h>

#include <string.h>

typedef struct {
    int numRows;
    int numCols;
    int nonZeroCount;
    double* values;
    int* colIndex;
    int* rowPtr;
} CSR;

CSR convertToCSR(double** matrix, int numRows, int numCols) {
    int nonZeroCount = 0;
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            if (matrix[i][j] != 0) {
                nonZeroCount++;
            }
        }
    }

    double* values = (double*)malloc(nonZeroCount * sizeof(double));
    int* colIndex = (int*)malloc(nonZeroCount * sizeof(int));
    int* rowPtr = (int*)malloc((numRows + 1) * sizeof(int));

    int k = 0;
    for (int i = 0; i < numRows; i++) {
        rowPtr[i] = k;
        for (int j = 0; j < numCols; j++) {
            if (matrix[i][j] != 0) {
                values[k] = matrix[i][j];
                colIndex[k] = j;
                k++;
            }
        }
    }
    rowPtr[numRows] = nonZeroCount;

    CSR csrMatrix = {numRows, numCols, nonZeroCount, values, colIndex, rowPtr};
    return csrMatrix;
}

void freeCSR(CSR csr){
    free(csr.values);
    free(csr.colIndex);
    free(csr.rowPtr);
}

void rowBlockParallelPTP(CSR csrMatrix, double** vector, double* result, int startRow, int endRow, int vectorCols) {
    for (int i = startRow; i < endRow; i++) {
        for (int k = 0; k < vectorCols; k++) {
            result[(i - startRow) * vectorCols + k] = 0.0;
            for (int j = csrMatrix.rowPtr[i]; j < csrMatrix.rowPtr[i + 1]; j++) {
                result[(i - startRow) * vectorCols + k] += csrMatrix.values[j] * vector[csrMatrix.colIndex[j]][k];
            }
        }
    }
}

void generateSparseMatrix(double** matrix, int numRows, int numCols, int nonZeroCount) {
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            matrix[i][j] = 0.0;
        }
    }

    srand((unsigned)time(NULL));
    for (int i = 0; i < nonZeroCount; ++i) {
        int row = rand() % numRows;
        int col = rand() % numCols;
        matrix[row][col] = (double)(rand() % 100);
    }
}

void generateVector(double** vector, int numRows, int numCols) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            vector[i][j] = (double)(rand() % 100);
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int numRows = 20000, numCols = 20000, vectorCols = 5;
    const int nonZeroCount = 20000000;

    double** matrix = (double**)malloc(numRows * sizeof(double*));
    for (int i = 0; i < numRows; ++i) {
        matrix[i] = (double*)malloc(numCols * sizeof(double));
    }
    generateSparseMatrix(matrix, numRows, numCols, nonZeroCount);

    double** vector = (double**)malloc(numRows * sizeof(double*));
    for (int i = 0; i < numRows; i++) {
        vector[i] = (double*)malloc(vectorCols * sizeof(double));
    }
    double* localResult = (double*)malloc(numRows * vectorCols * sizeof(double));
    if (rank == 0) {
        generateSparseMatrix(matrix, numRows, numCols, nonZeroCount);
        generateVector(vector, numRows, vectorCols);
    }

    CSR csrMatrix = convertToCSR(matrix, numRows, numCols);

    double start_time = MPI_Wtime();

    int rowsPerProc = numRows / size;
    int startRow = rank * rowsPerProc;
    int endRow = (rank == size - 1) ? numRows : startRow + rowsPerProc;
    rowBlockParallelPTP(csrMatrix, vector, localResult, startRow, endRow, vectorCols);

    double* globalResult = NULL;
    if (rank == 0) {
        globalResult = (double*)malloc(numRows * sizeof(double));
        memcpy(globalResult, localResult, rowsPerProc * sizeof(double));
        for (int i = 1; i < size; i++) {
            MPI_Recv(globalResult + i * rowsPerProc, rowsPerProc, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(localResult, rowsPerProc, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("From Rank %d:\n", rank);
        printf("Communication time: %f seconds\n", end_time - start_time);
        int printCount = 0;
        for (int i = 0; i < numRows; i++) {
            for (int k = 0; k < vectorCols; k++) {
                if (globalResult[i * vectorCols + k] != 0) {
                    printf("Result[%d][%d] = %f\n", i, k, globalResult[i * vectorCols + k]);
                    printCount++;
                    if (printCount >= 20) {
                        break;
                    }
                }
            }
            if (printCount >= 20) {

                break;
            }
        }
    }

    free(matrix);
    free(vector);
    free(localResult);
    if (rank == 0) {
        free(globalResult);
    }
    freeCSR(csrMatrix);
    MPI_Finalize();
    return 0;
}



