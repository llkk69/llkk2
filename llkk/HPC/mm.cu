#include <stdio.h>
#include <cuda.h>

__global__ void matadd(int *l, int *m, int *n) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int id = gridDim.x * y + x;
    n[id] = l[id] + m[id];
}

int main() {
    int a[2][3];
    int b[2][3];
    int c[2][3];
    int *d, *e, *f;
    int i, j;

    // Input for matrix A
    printf("Enter elements of first matrix of size 2 * 3:\n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 3; j++) {
            scanf("%d", &a[i][j]);
        }
    }

    // Input for matrix B
    printf("Enter elements of second matrix of size 2 * 3:\n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 3; j++) {
            scanf("%d", &b[i][j]);
        }
    }

    // Allocate device memory
    cudaMalloc((void **)&d, 2 * 3 * sizeof(int));
    cudaMalloc((void **)&e, 2 * 3 * sizeof(int));
    cudaMalloc((void **)&f, 2 * 3 * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(d, a, 2 * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(e, b, 2 * 3 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(3, 2);

    // Measure elapsed time
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch kernel for matrix addition
    matadd<<<grid, 1>>>(d, e, f);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed Time: %.6f ms\n", elapsedTime);

    // Copy the result matrix from device to host
    cudaMemcpy(c, f, 2 * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("\nSum of two matrices:\n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 3; j++) {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d);
    cudaFree(e);
    cudaFree(f);

    return 0;
}
