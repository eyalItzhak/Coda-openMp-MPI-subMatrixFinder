#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>
#include "myProto.h"

// This function is a special - runs on Host and Device
__device__ __host__ float dis_func(int pic_num, int obj_num)
{
    double myRes = double(double(pic_num - obj_num) / double(pic_num));
    // printf("%f from gpu \n",myRes);
    double res = fabs(myRes);
    return (float)res;
}

__global__ void kernel(int **pic, int **obj, float *res, int dimMatrix, int offsetRow, int offsetCol)
{                                                  // get matrix pic matirx obj and res for output result=>pic get offsets that simbol the stat of sub image inside pic.
    int i = blockDim.x * blockIdx.x + threadIdx.x; // witch itration are we =>need to be of size dim_obj*dim_obj =>total elements of object we want to compare...
    int row = i / dimMatrix;                       // the row of this specific element
    int colum = i % dimMatrix;                     // the colum of this specific element
    //  printf("my val %d =>form gpu\n",pic[row+offsetRow][colum+offsetCol]);
    if (i < dimMatrix * dimMatrix)
        res[i] = dis_func(pic[row + offsetRow][colum + offsetCol], obj[row][colum]); // take the value from pic from location of the sub image we want to compare to
}

int computeOnGPU(int **picture, int **object, int dim_pic, int dim_obj, float threshold)
{

    cudaError_t err = cudaSuccess;
    //***********************************
    // the pic array
    int **d_A; // will contain list of arrays
    size_t size = dim_pic * sizeof(int *);

    cudaMalloc((void ***)&d_A, size);

    int **d_A_array = (int **)malloc(dim_pic * sizeof(int *)); // will contain the first array
    for (int i = 0; i < dim_pic; i++)
    {
        cudaMalloc((void **)&(d_A_array[i]), dim_pic * sizeof(int));                         // new mem on the gpu
        cudaMemcpy(d_A_array[i], picture[i], dim_pic * sizeof(int), cudaMemcpyHostToDevice); // put the actual value inside
    }
    cudaMemcpy(d_A, d_A_array, dim_pic * sizeof(int *), cudaMemcpyHostToDevice);

    //***********************************
    // the pic array
    int **d_B; // will contain list of arrays
    cudaMalloc((void ***)&d_B, dim_obj * sizeof(int *));

    int **d_B_array = (int **)malloc(dim_obj * sizeof(int *)); // will contain the first array
    for (int i = 0; i < dim_obj; i++)
    {
        cudaMalloc((void **)&(d_B_array[i]), dim_obj * sizeof(int));                        // new mem on the gpu
        cudaMemcpy(d_B_array[i], object[i], dim_obj * sizeof(int), cudaMemcpyHostToDevice); // put the actual value inside
    }
    cudaMemcpy(d_B, d_B_array, dim_obj * sizeof(int *), cudaMemcpyHostToDevice);

    //***********************************
    // the res array
    size = dim_obj * dim_obj * sizeof(float);
    float *d_C;
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *C = (float *)malloc(size);
    //***********************************
    // set up the gpu for the run
    int threadsPerBlock = 256;
    int blocksPerGrid = ((dim_obj * dim_obj) + threadsPerBlock - 1) / threadsPerBlock;
    // use the gpu

    int itrations = dim_pic - dim_obj;
    float res = 0;
    int boolres = 0;
    for (int row = 0; row < itrations; row++)
    {
        for (int col = 0; col < itrations; col++)
        {
            kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, dim_obj, row, col);

            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch vectorAdd kernel -  %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // get info after run
            err = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy result array from device to host -%s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            for (int i = 0; i < dim_obj * dim_obj; i++)
            {
                res = res + C[i];
            }
            if (res <= threshold)
            {
                printf("location {%d , %d}", row, col);
                boolres = 1;
                row = itrations;
                col = itrations;
            }
            res = 0;
        }
    }

    if (cudaFree(d_A) != cudaSuccess || cudaFree(d_B) != cudaSuccess || cudaFree(d_C) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < dim_pic; i++)
    {
        if (cudaFree(d_A_array[i]) != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < dim_obj; i++)
    {
        // cudaMalloc((void**) &(d_B_array[i]), dim_obj*sizeof(int));
        if (cudaFree(d_B_array[i]) != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    return boolres;
}