
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <windows.h>

/**
* @Brief Cuda Implementation to parallelize the matrix addition
* 
* The addition is performed by threads in the gpu cores and the id of the
* threads are used for indexing the matrices. For simplicity, it is
* assumed that the implementation of the matrices are using 1d array(vector).
* 
* @param a matrix a which is used in addition
* @param b matrix b which is used in addition
* @param c matrix c which is used to store the result of addition
* @param m row number of the matrices
* @param n col number of the matrices
*/
__global__ void VecAdd(float* a, float* b, float* c, int m, int n)
{
    // Multiply the Id of the block with total threads in a block + the current thread Id to get the right index.
    int ind = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (ind < m * n) {
        c[ind] = a[ind] + b[ind];
    }
}

/**
* @Brief Cuda Implementation to parallelize the matrix multiplication
* 
* The multiplication is performed by threads in the gpu cores and the id of the
* threads are used for indexing the matrices. For simplicity, no coelascing memory
* is used and the gpu warps are taken to be 2d-array instead of 1d-array. However,
* the matrices are treated as 1d-array in the memory. It is also assumed that
* the 2 matrices are valid for multiplication and have the same shape.
* 
* @param a matrix a which is used in multiplication
* @param b matrix b which is used in multiplication
* @param c matrix c which is used to store the result
* @param row num of row in the matrices
* @param col num of col in the matrices
*/
__global__ void parallel_mult(float* a, float* b, float* c, int row, int col) {
    int row_i = blockIdx.y * blockDim.y + threadIdx.y;
    int col_j = blockIdx.x * blockDim.x + threadIdx.x;

    // Check to make sure extra threads don't need to do anything.
    if (row_i < row && col_j < col) {
        c[row_i * col + col_j] = 0.0f;
        // Add every term for the row to the entire column
        for (int i = 0; i < col; i++) {
            c[row_i * col + col_j] += a[row_i * col + i] * b[i * col + col_j];
        }
    }
}

/*
The following implementation is inspired by Girish Sharma paper on how we can
break down the steps in the elimination to form a parallel implementation.
*/
/**
* @Brief This function is to perform normalization on the entries
* which are not on the diagonal
* 
* This is 1 step inside Gaussian Jordan elimination and could be performed
* as parallel task. Similarly, we treated the gpu warps as 2d-array while
* the actual implementation of the matrices as 1d-array. It is assumed that
* the matrix a is a non-singular square matrix for simplicity.
* 
* @param a matrix a which is used to find the inverse
* @param I identity matrix which is used to store the inverse matrix
* @param col dimension of the matrix
* @param row current row that operations are undertaken
*/
__global__ void non_diag_norm(float* a, float* I, int col, int row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < col && y < col) {
        float factor = a[row * col + row];

        if (x == row && x != y) {
            I[x * col + y] /= factor;
            a[x * col + y] /= factor;
        }
    }
}

/**
* @Brief Perform normalization on the diagonal entries
* 
* This is another step in Gaussian Jordan elimination and could be
* performed as parallel task. Similarly, we treated gpu warps as 2d-array while
* the actual implementation of the matrices as 1d-array. It is assumed that
* the matrix a is a non-singular square matrix for simplicity.
* 
* @param a matrix a which is used to find the inverse
* @param I identity matrix which is used to store the inverse matrix
* @param col dimension of the matrix
* @param row current row that operations are undertaken
*/
__global__ void diag_norm(float* a, float* I, int col, int row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < col && y < col) {
        float factor = a[row * col + row];

        if (x == y && x == row) {
            I[x * col + y] /= factor;
            a[x * col + y] /= factor;
        }
    }
}

/**
* @Brief Perform row operations on the other rows that are not pivot row
*
* This is another step in Gaussian Jordan elimination and could be
* performed as parallel task. Similarly, we treated gpu warps as 2d-array while
* the actual implementation of the matrices as 1d-array. It is assumed that
* the matrix a is a non-singular square matrix for simplicity.
*
* @param a matrix a which is used to find the inverse
* @param I identity matrix which is used to store the inverse matrix
* @param col dimension of the matrix
* @param row current row that operations are undertaken
*/
__global__ void gaussian(float* a, float* I, int col, int row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < col && y < col) {
        if (x != row) {
            I[x * col + y] -= I[row * col + y] * a[x * col + row];

            if (y != row) {
                a[x * col + y] -= a[row * col + y] * a[x * col + row];
            }
        }
    }
}

/**
* @Brief Set the elements below the pivot element to be 0
*
* @param a matrix a which is used to find the inverse
* @param I identity matrix which is used to store the inverse matrix
* @param col dimension of the matrix
* @param row current row that operations are undertaken
*/
__global__ void zero(float* a, float* I, int col, int row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < col && y < col) {
        if (x != row) {
            if (y == row) {
                a[x * col + y] = 0;
            }
        }
    }
}


/**
* @Brief Function to perform analysis on matrix addition
* 
* @param flag determine whether to run serial code or parallel code
* @param rounds number of iterations for averaging
*/
void testAdd(char flag, int rounds) {
    // Initial Setup
    cudaSetDevice(0);
    srand(42);

    // Display for Test
    if (flag == 'g') {
        printf("Start Test on GPU\n");
    }

    // Initial configuration
    int sizes[] = { 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000};
    LARGE_INTEGER frequency, start, end, comp_start, comp_end, for_start, for_end;
    double total_duration = 0, comp_duration = 0, for_duration = 0;

    QueryPerformanceFrequency(&frequency);
    for (int i = 0; i < sizeof(sizes) / sizeof(int); i++) {
        // printf("Test with size %d\n", sizes[i]);
        QueryPerformanceCounter(&start);

        // Allocate memory
        int size = sizes[i] * sizes[i];
        float* a = (float*)malloc(size * sizeof(float));
        float* b = (float*)malloc(size * sizeof(float));
        float* c = (float*)calloc(size, sizeof(float));

        // Random matrix generation
        for (int j = 0; j < size; j++) {
            a[j] = (float)(rand() % 100000);
            b[j] = (float)(rand() % 100000);
        }

        // GPU CUDA Compute
        if (flag == 'g') {
            // Cuda memory allocation
            float* _a, * _b, * _c;
            cudaMalloc(&_a, size * sizeof(float));
            cudaMalloc(&_b, size * sizeof(float));
            cudaMalloc(&_c, size * sizeof(float));

            cudaMemcpy(_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

            // Calculate gpu threads and blocks
            int numThreads = 1024;
            int numBlocks = (sizes[i] + numThreads - 1) / numThreads;

            // Compute the result
            QueryPerformanceCounter(&for_start);
            for (int j = 0; j < rounds; j++) {
                QueryPerformanceCounter(&comp_start);
                VecAdd<<<numBlocks, numThreads>>>(_a, _b, _c, sizes[i], sizes[i]);
                // Wait for all the threads to finish execute
                cudaDeviceSynchronize();
                QueryPerformanceCounter(&comp_end);

                comp_duration += (double)(comp_end.QuadPart - comp_start.QuadPart) / frequency.QuadPart;
            }
            QueryPerformanceCounter(&for_end);

            // Return the result to host
            cudaMemcpy(c, _c, size * sizeof(float), cudaMemcpyDeviceToHost);
            // Freeing memory in GPU
            cudaFree(_a);
            cudaFree(_b);
            cudaFree(_c);
        }

        free(a);
        free(b);
        free(c);
        QueryPerformanceCounter(&end);

        // Time calculation
        for_duration = (double)(for_end.QuadPart - for_start.QuadPart) / frequency.QuadPart;
        total_duration = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
        total_duration = total_duration - for_duration + (for_duration / rounds);
        comp_duration = comp_duration / rounds;
        // printf("Computation Duration : %f\n", comp_duration);
        // printf("Total Duration : %f\n\n", total_duration);
        printf("%d, %f\n", sizes[i], comp_duration);
    }
    return;
    cudaDeviceReset();
}

/**
* @Brief Function to perform analysis on matrix multiplication
*
* @param flag determine whether to run serial code or parallel code
* @param rounds number of iterations for averaging
*/
void testMult(char flag, int rounds) {
    // Initial setup
    cudaSetDevice(0);
    srand(42);

    if (flag == 'g') {
        printf("Start Test on GPU\n");
    }

    // Initial configuration
    int sizes[] = { 2, 5, 10, 50, 100, 500, 1000, 2000};
    LARGE_INTEGER frequency, start, end, comp_start, comp_end, for_start, for_end;
    double total_duration = 0, comp_duration = 0, for_duration = 0;

    QueryPerformanceFrequency(&frequency);
    for (int i = 0; i < sizeof(sizes) / sizeof(int); i++) {
        QueryPerformanceCounter(&start);

        // Allocate memory
        int size = sizes[i] * sizes[i];
        float* a = (float*)malloc(size * sizeof(float));
        float* b = (float*)malloc(size * sizeof(float));
        float* c = (float*)calloc(size, sizeof(float));

        // Random matrix generation
        for (int j = 0; j < size; j++) {
            a[j] = (float)(rand() % 1000);
            b[j] = (float)(rand() % 1000);
        }

        if (flag == 'g') {
            // cuda memory allocation
            float* _a, * _b, * _c;
            cudaMalloc(&_a, size * sizeof(float));
            cudaMalloc(&_b, size * sizeof(float));
            cudaMalloc(&_c, size * sizeof(float));

            cudaMemcpy(_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
            
            // Threads calculation
            int threads = 1024;
            dim3 numThreads(threads, threads);
            dim3 numBlocks((sizes[i] + threads - 1) / threads, (sizes[i] + threads - 1) / threads);

            // Computation
            QueryPerformanceCounter(&for_start);
            for (int j = 0; j < rounds; j++) {
                QueryPerformanceCounter(&comp_start);
                parallel_mult<<<numBlocks, numThreads>>>(_a, _b, _c, sizes[i], sizes[i]);
                // Wait for GPU to finish executing
                cudaDeviceSynchronize();
                QueryPerformanceCounter(&comp_end);

                comp_duration += (double)(comp_end.QuadPart - comp_start.QuadPart) / frequency.QuadPart;
            }
            QueryPerformanceCounter(&for_end);

            // Return result to host
            cudaMemcpy(c, _c, size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(_a);
            cudaFree(_b);
            cudaFree(_c);
        }

        free(a);
        free(b);
        free(c);
        QueryPerformanceCounter(&end);

        // Time calculation
        for_duration = (double)(for_end.QuadPart - for_start.QuadPart) / frequency.QuadPart;
        total_duration = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
        total_duration = total_duration - for_duration + (for_duration / rounds);
        comp_duration = comp_duration / rounds;
        printf("%d, %f\n", sizes[i], comp_duration);
        // printf("Computation Duration : %f\n", comp_duration);
        // printf("Total Duration : %f\n\n", total_duration);
    }
    return;
    cudaDeviceReset();
}

/**
* @Brief Function to perform analysis on matrix inversion
*
* @param flag determine whether to run serial code or parallel code
* @param rounds number of iterations for averaging
*/
void testInv(char flag, int rounds) {
    // Initial setup
    cudaSetDevice(0);
    srand(42);

    if (flag == 'g') {
        printf("Start Test on GPU\n");
    }

    // Initial configuration
    int sizes[] = { 2, 5, 10, 50, 100, 500, 1000 };
    LARGE_INTEGER frequency, start, end, comp_start, comp_end, for_start, for_end, cpy_start, cpy_end;
    double total_duration = 0, comp_duration = 0, for_duration = 0, cpy_duration = 0;

    QueryPerformanceFrequency(&frequency);
    for (int i = 0; i < sizeof(sizes) / sizeof(int); i++) {
        // printf("Test with size %d \n", sizes[i]);
        QueryPerformanceCounter(&start);

        // Memory allocation
        int size = sizes[i] * sizes[i];
        float* a = (float*)malloc(size * sizeof(float));
        float* b = (float*)calloc(size, sizeof(float));

        // Random matrix generation
        for (int j = 0; j < size; j++) {
            a[j] = (float)(rand() % 10000);
        }

        // Identity matrix generation
        for (int j = 0; j < sizes[i]; j++) {
            a[j * sizes[i] + j] = 1.0f;
        }

        if (flag == 'g') {
            // Memory allocation on cuda
            float* _a, * _b;
            cudaMalloc(&_a, size * sizeof(float));
            cudaMalloc(&_b, size * sizeof(float));

            cudaMemcpy(_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

            // threads calculation
            int threads = 1024;
            dim3 numThreads(threads, threads);
            dim3 numBlocks((sizes[i] + threads - 1) / threads, (sizes[i] + threads - 1) / threads);

            // Computations
            QueryPerformanceCounter(&for_start);
            for (int j = 0; j < rounds; j++) {
                QueryPerformanceCounter(&comp_start);
                for (int k = 0; k < sizes[i]; k++) {
                    non_diag_norm<<<numBlocks, numThreads>>>(_a, _b, sizes[i], k);
                    diag_norm<<<numBlocks, numThreads>>>(_a, _b, sizes[i], k);
                    gaussian<<<numBlocks, numThreads>>>(_a, _b, sizes[i], k);
                    zero<<<numBlocks, numThreads>>>(_a, _b, sizes[i], k);
                }
                cudaDeviceSynchronize();
                QueryPerformanceCounter(&comp_end);

                comp_duration += (double)(comp_end.QuadPart - comp_start.QuadPart) / frequency.QuadPart;

                QueryPerformanceCounter(&cpy_start);
                if (j < rounds - 1) {
                    cudaMemcpy(_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
                }
                QueryPerformanceCounter(&cpy_end);

                cpy_duration += (double)(cpy_end.QuadPart - cpy_start.QuadPart) / frequency.QuadPart;
            }
            QueryPerformanceCounter(&for_end);

            // Return result to host
            cudaMemcpy(b, _b, size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(_a);
            cudaFree(_b);
        }

        free(a);
        free(b);
        QueryPerformanceCounter(&end);

        // Time calculation
        for_duration = (double)(for_end.QuadPart - for_start.QuadPart) / frequency.QuadPart;
        for_duration += cpy_duration / (rounds-1);
        total_duration = (double)(end.QuadPart - start.QuadPart) / frequency.QuadPart;
        total_duration = total_duration - for_duration + (for_duration / rounds);
        comp_duration = comp_duration / rounds;
        // printf("Computation Duration : %f\n", comp_duration);
        // printf("Total Duration : %f\n\n", total_duration);
        printf("%d, %f\n", sizes[i], comp_duration);
    }

    return;
    cudaDeviceReset();
}

int main(int argc, char** argv) {
    // Run the test

    printf("Performing Addition\n");
    testAdd('g', 10);

    printf("Performing Multiplication\n");
    testMult('g', 10);

    printf("Performing Inversion\n");
    testInv('g', 10);

    // shut off the machine
    cudaDeviceReset();
    return 0;
}
