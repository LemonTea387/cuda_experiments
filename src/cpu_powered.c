
#include <stdio.h>
#include <windows.h>
#include <omp.h>

const int  NUM_THREADS = 8;

/**
* @Brief Perform matrix addition in serial
* 
* This is addition of matrix run in serial using a for loop
* 
* @param a matrix a which is used in addition
* @param b matrix b which is used in addition
* @param c matrix c which is used to store the result of addition
* @param m row number of the matrices
* @param n col number of the matrices
*/
void vec_add_serial(float* a, float* b, float* c, int m, int n)
{
    for (int i = 0; i < m * n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

/**
* @Brief Serial Implementation of matrix multiplication
*
* @param a matrix a which is used in multiplication
* @param b matrix b which is used in multiplication
* @param c matrix c which is used to store the result
* @param row num of row in the matrices
* @param col num of col in the matrices
*/
void serial_mult(float* a, float* b, float* c, int row, int col) {
    int n = row * col;
    for (int i = 0; i < n; i++) {
        c[i] = 0;
        for (int j = 0; j < col; j++) {
            c[i] += a[(i / col) * col + j] * b[(i % row) + j * row];
        }
    }
}

/**
* @Brief This is serial implementation of Gaussian Jordan Algorithm
* 
* @param a matrix a which is used to find inverse
* @param I identity matrix for storing the inverse
* @param row row number of the matrix
* @param col col number of the matrix
*/
void serial_inv(float* a, float* I, int row, int col) {
    int pivot;
    float factor;

    for (int i = 0; i < row; i++) {
        pivot = i;

        // Get the largest pivot
        for (int j = i + 1; j < row; j++) {
            if (a[j * col + i] > a[j * col + pivot])
                pivot = j;
        }

        // Swap Rows for pivot row
        if (pivot != i) {
            float temporary;

            for (int j = 0; j < col; j++) {
                temporary = a[i * col + j];
                a[i * col + j] = a[pivot * col + j];
                a[pivot * col + j] = temporary;

                temporary = I[i * col + j];
                I[i * col + j] = I[pivot * col + j];
                I[pivot * col + j] = temporary;
            }
        }

        // Perform row eliminations
        for (int j = 0; j < row; j++) {
            if (i != j) {
                factor = a[j * col + i];

                for (int k = 0; k < col; k++) {
                    a[j * col + k] -= (a[i * col + k] / a[i * col + i]) * factor;
                    I[j * col + k] -= (I[i * col + k] / a[i * col + i]) * factor;
                }
            }
            else {
                factor = a[j * col + i];

                for (int k = 0; k < col; k++) {
                    a[j * col + k] /= factor;
                    I[j * col + k] /= factor;
                }
            }
        }
    }
}

/**
 * @brief CPU Parallelism using OpenMP for addition of float vector a and b into result float vector c with dimension row x col.
 * It is assumed that it works on 1d array(vector), with the concept of row and col to give the illusion of 2d array(matrix).
 * The result is written into result float vector c.
 * 
 * @param a Pointer to float vector a
 * @param b Pointer to float vector b
 * @param c Pointer to float vector c
 * @param row Dimension row
 * @param col Dimension col
 */
void parallel_cpu_add(float* a, float* b, float* c, int row, int col) {
    #pragma omp parallel for shared(a, b, c)
    for (int i = 0; i < row * col; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief CPU parallelism using OpenMP for multiplication of float vector a and b into result float vector c with dimension row x col.
 * Accumulates by reduction for the elements of the column as per ordinary matrix multiplication method of row of matrix A * column of matrix B.
 * 
 * @param a Pointer to float vector a
 * @param b Pointer to float vector b
 * @param c Pointer to float vector c
 * @param row Dimension row
 * @param col Dimension col
 */
void parallel_cpu_mult(float* a, float* b, float* c, int row, int col) {
    #pragma omp parallel for shared(c)
    for (int i = 0; i < row * col; i++) {
        c[i] = 0;
    }
    int size = row * col;
    #pragma omp parallel for collapse(2) reduction(+:c[:size]) shared(a, b)
    for (int i = 0; i < row * col; i++) {
        for (int j = 0; j < col; j++) {
            // printf("Size: %d Index a : %d ,b: %d\n", row*col,(i / col) * col + j, (i % row) + j * row);
            c[i] += a[(i / col) * col + j] * b[(i % row) + j * row];
        }
    }
}

/**
 * @brief CPU Parallellism using OpenMP for Inversing float vector a into result float vector I with dimension row x col.
 * 
 * @param a Pointer to float vector a
 * @param I Pointer to inverse float vector I
 * @param row Dimension row
 * @param col Dimension col
 */
void parallel_cpu_inv(float* a, float* I, int row, int col) {
    for (int i = 0; i < row; i++) {
        // Normalize the current row i
        #pragma omp parallel for shared(a, I, i)
        for (int j = 0; j < col; j++) {
            if (j != i) {
                a[i * col + j] /= a[i * col + i];
                I[i * col + j] /= a[i * col + i];
            }
        }

        // Normalize the diagonal / pivot element on current row i
        I[i * col + i] /= a[i * col + i];
        a[i * col + i] /= a[i * col + i];

        // Run row operations on other rows
        #pragma omp parallel for collapse(2) shared(a, I, i)
        for (int j = 0; j < row; j++) {
            for (int k = 0; k < col; k++) {
                if (j != i) {
                    I[j * col + k] -= I[i * col + k] * a[j * col + i];

                    if (k != i) {
                        a[j * col + k] -= a[i * col + k] * a[j * col + i];
                    }
                }
            }
        }

        // Set element below pivot to be 0
        #pragma omp parallel for shared(a, I, i)
        for (int j = 0; j < row; j++) {
            if (j != i) {
                a[j * col + i] = 0;
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
    // initial setup
    srand(42);

    // Display for Test
    if (flag == 'c') {
        printf("Start Test on Parallel\n");
    }
    else if (flag == 's') {
        printf("Start Test on Serial\n");
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

        // CPU Parallelism
        if (flag == 'c') {
            omp_set_num_threads(NUM_THREADS);

            QueryPerformanceCounter(&for_start);
            for (int j = 0; j < rounds; j++) {
                QueryPerformanceCounter(&comp_start);
                parallel_cpu_add(a, b, c, sizes[i], sizes[i]);
                QueryPerformanceCounter(&comp_end);
                comp_duration += (double)(comp_end.QuadPart - comp_start.QuadPart) / frequency.QuadPart;
            }
            QueryPerformanceCounter(&for_end);
        
        }
        // Serial Addition
        else if (flag == 's') {
            QueryPerformanceCounter(&for_start);
            for (int j = 0; j < rounds; j++) {
                QueryPerformanceCounter(&comp_start);
                vec_add_serial(a, b, c, sizes[i], sizes[i]);
                QueryPerformanceCounter(&comp_end);
                comp_duration += (double)(comp_end.QuadPart - comp_start.QuadPart) / frequency.QuadPart;
            }
            QueryPerformanceCounter(&for_end);
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
}

/**
* @Brief Function to perform analysis on matrix multiplication
*
* @param flag determine whether to run serial code or parallel code
* @param rounds number of iterations for averaging
*/
void testMult(char flag, int rounds) {
    // Initial setup
    srand(42);

    if (flag == 'c') {
        printf("Start Test on Parallel\n");
    }
    else if (flag == 's') {
        printf("Start Test on Serial\n");
    }

    // Initial configuration
    int sizes[] = { 2, 5, 10, 50, 100, 500, 700};
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
            a[j] = (float)(rand() % 1000);
            b[j] = (float)(rand() % 1000);
        }

        if (flag == 'c') {
            omp_set_num_threads(NUM_THREADS);

            QueryPerformanceCounter(&for_start);
            for (int j = 0; j < rounds; j++) {
                QueryPerformanceCounter(&comp_start);
                parallel_cpu_mult(a, b, c, sizes[i], sizes[i]);
                QueryPerformanceCounter(&comp_end);

                comp_duration += (double)(comp_end.QuadPart - comp_start.QuadPart) / frequency.QuadPart;
            }
            QueryPerformanceCounter(&for_end);
        }
        else if (flag == 's') {
            // Serial running
            QueryPerformanceCounter(&for_start);
            for (int j = 0; j < rounds; j++) {
                QueryPerformanceCounter(&comp_start);
                serial_mult(a, b, c, sizes[i], sizes[i]);
                QueryPerformanceCounter(&comp_end);

                comp_duration += (double)(comp_end.QuadPart - comp_start.QuadPart) / frequency.QuadPart;
            }
            QueryPerformanceCounter(&for_end);
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
}

/**
* @Brief Function to perform analysis on matrix inversion
*
* @param flag determine whether to run serial code or parallel code
* @param rounds number of iterations for averaging
*/
void testInv(char flag, int rounds) {
    // Initial setup
    srand(42);

    if (flag == 'c') {
        printf("Start Test on Parallel\n");
    }
    else if (flag == 's') {
        printf("Start Test on Serial\n");
    }

    // Initial configuration
    int sizes[] = { 2, 5, 10, 50, 100, 500, 1000 };
    LARGE_INTEGER frequency, start, end, comp_start, comp_end, for_start, for_end, cpy_start, cpy_end;
    double total_duration = 0.0, comp_duration = 0.0, for_duration = 0.0, cpy_duration = 0.0;

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

        if (flag == 'c') {
            omp_set_num_threads(NUM_THREADS);

            QueryPerformanceCounter(&for_start);
            for (int j = 0; j < rounds; j++) {
                QueryPerformanceCounter(&comp_start);
                parallel_cpu_inv(a, b, sizes[i], sizes[i]);
                QueryPerformanceCounter(&comp_end);

                comp_duration += (double)(comp_end.QuadPart - comp_start.QuadPart) / frequency.QuadPart;

                QueryPerformanceCounter(&cpy_start);
                if (j < rounds - 1) {
                    for (int j = 0; j < size; j++) {
                        a[j] = (float)(rand() % 10000);
                    }

                    int counter = 0;
                    for (int j = 0; j < size; j++) {
                        if (j / sizes[i] == 0 && j % sizes[i] == counter) 
                        {
                            a[j] = 1.0f;
                            counter++;
                        }
                        else 
                        {
                            a[j] = 0.0f;
                        }
                    }
                }
                QueryPerformanceCounter(&cpy_end);

                cpy_duration += (double)(cpy_end.QuadPart - cpy_start.QuadPart) / frequency.QuadPart;
            }
            QueryPerformanceCounter(&for_end);
        }
        else if (flag == 's') {
            // Serial computation
            QueryPerformanceCounter(&for_start);
            for (int j = 0; j < rounds; j++) {
                QueryPerformanceCounter(&comp_start);
                serial_inv(a, b, sizes[i], sizes[i]);
                QueryPerformanceCounter(&comp_end);

                comp_duration += (double)(comp_end.QuadPart - comp_start.QuadPart) / frequency.QuadPart;
                
                QueryPerformanceCounter(&cpy_start);
                if (j < rounds - 1) {
                    for (int j = 0; j < size; j++) {
                        a[j] = (float)(rand() % 10000);
                    }

                    int counter = 0;
                    for (int j = 0; j < size; j++) {
                        if (j / sizes[i] == 0 && j % sizes[i] == counter) 
                        {
                            a[j] = 1.0f;
                            counter++;
                        }
                        else 
                        {
                            a[j] = 0.0f;
                        }
                    }
                }
                QueryPerformanceCounter(&cpy_end);

                cpy_duration += (double)(cpy_end.QuadPart - cpy_start.QuadPart) / frequency.QuadPart;
            }
            QueryPerformanceCounter(&for_end);
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
}

int main(int argc, char** argv) {
    // Run the test
    printf("Performing Addition\n");
    testAdd('s', 10);
    testAdd('c', 10);

    printf("Performing Multiplication\n");
    testMult('s', 10);
    testMult('c', 10);

    printf("Performing Inversion\n");
    testInv('s', 10);
    testInv('c', 10);

    return 0;
}