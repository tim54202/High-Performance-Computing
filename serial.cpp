#include <mpi.h>   // Include MPI library for parallel computing.
#include <stdio.h> // Include standard I/O functions.
#include <stdlib.h> // Include standard library for memory allocation, rand, etc.
#include <time.h>  // Include time library for time functions.

// Define a structure for CSR representation of a sparse matrix.
typedef struct {
    int numRows;       // Number of rows in the matrix.
    int numCols;       // Number of columns in the matrix.
    int nonZeroCount;  // Number of non-zero elements in the matrix.
    double* values;    // Array to store the non-zero values.
    int* colIndex;     // Array to store the column indices of the non-zero values.
    int* rowPtr;       // Array to store the starting index of each row in 'values'.
} CSR;

// Function to convert a dense matrix to CSR format.
CSR convertToCSR(double** matrix, int numRows, int numCols) {
    // Count the number of non-zero elements in the matrix.
    int nonZeroCount = 0;
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            if (matrix[i][j] != 0) {
                nonZeroCount++;
            }
        }
    }

    // Allocate memory for CSR arrays.
    double* values = (double*)malloc(nonZeroCount * sizeof(double));
    int* colIndex = (int*)malloc(nonZeroCount * sizeof(int));
    int* rowPtr = (int*)malloc((numRows + 1) * sizeof(int));

    // Fill the CSR arrays with appropriate values.
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

    // Return the CSR structure.
    CSR csrMatrix = {numRows, numCols, nonZeroCount, values, colIndex, rowPtr};
    return csrMatrix;
}

// Function to free memory allocated for CSR matrix.
void freeCSR(CSR csr){
    free(csr.values);
    free(csr.colIndex);
    free(csr.rowPtr);
}

// Serial implementation of matrix-vector multiplication using CSR.
void serialAlgorithm(CSR csrMatrix, double** vector, double* result, int vectorCols) {
    for (int i = 0; i < csrMatrix.numRows; i++) {
        for (int k = 0; k < vectorCols; k++) {
            result[i * vectorCols + k] = 0;
            for (int j = csrMatrix.rowPtr[i]; j < csrMatrix.rowPtr[i + 1]; j++) {
                result[i * vectorCols + k] += csrMatrix.values[j] * vector[csrMatrix.colIndex[j]][k];
            }
        }
    }
}

// Function to generate a sparse matrix with random non-zero values.
void generateSparseMatrix(double** matrix, int numRows, int numCols, int nonZeroCount) {
    // Initialize all elements to zero.
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            matrix[i][j] = 0.0;
        }
    }

    // Randomly assign non-zero values.
    int filled = 0;
    while (filled < nonZeroCount) {
        int row = rand() % numRows;
        int col = rand() % numCols;
        if (matrix[row][col] == 0.0) {
            matrix[row][col] = (double)(rand() % 100);
            filled++;
        }
    }
}

// Function to generate a dense vector with random values.
void generateVector(double** vector, int numRows, int numCols) {
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            vector[i][j] = (double)(rand() % 100);
        }
    }
}

// Main function for the MPI program.
int main(int argc, char* argv[]) {
    // Initialize MPI environment.
    MPI_Init(&argc, &argv);

    // Get the rank and size in the MPI communicator.
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define matrix and vector dimensions.
    const int numRows = 20000, numCols = 20000, vectorCols = 5;
    const int nonZeroCount = 20000000;

    // Allocate memory for matrix and vector.
    double** matrix = (double**)malloc(numRows * sizeof(double*));
    for (int i = 0; i < numRows; ++i) {
        matrix[i] = (double*)malloc(numCols * sizeof(double));
    }

    double** vector = (double**)malloc(numRows * sizeof(double*));
    for (int i = 0; i < numRows; ++i) {
        vector[i] = (double*)malloc(vectorCols * sizeof(double));
    }

    // Allocate memory for the result vector.
    double* result = (double*)malloc(numRows * vectorCols * sizeof(double));

    // Master process generates the matrix and vector, and performs the multiplication.
    if (rank == 0) {
        generateSparseMatrix(matrix, numRows, numCols, nonZeroCount);
        generateVector(vector, numRows, vectorCols);

        CSR csrMatrix = convertToCSR(matrix, numRows, numCols);

        double start_time = MPI_Wtime();  // Start timing the computation.
        serialAlgorithm(csrMatrix, vector, result, vectorCols);
        double end_time = MPI_Wtime();    // End timing the computation.

        printf("From Rank %d:\n", rank);
        printf("Communication time: %f seconds\n", end_time - start_time);

        // Print a subset of the results.
        int printCount = 0;
        for (int i = 0; i < numRows; i++) {
            for (int k = 0; k < vectorCols; k++) {
                if (result[i * vectorCols + k] != 0) {
                    printf("Result[%d][%d] = %f\n", i, k, result[i * vectorCols + k]);
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
        freeCSR(csrMatrix); // Free CSR matrix memory.
    }

    // Free memory allocated for matrix and vector.
    for (int i = 0; i < numRows; ++i) {
        free(matrix[i]);
        free(vector[i]);
    }
    free(matrix);
    free(vector);
    free(result);

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}
