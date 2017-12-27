#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cassert>

#define ITERATIONS 10
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1
//#include "texture.h"
//#define THREADS_PER_BLOCK 32
using namespace std;

__global__ void spmv(int* num_col, float* val, float* vec, float* res, int dim, int max_col)

{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0;
    if(i<dim)
    {
      //#pragma unroll    //用了这个好像会变慢
      for(int j=0; j<max_col; j++){
          tmp+= val[j*dim+i] * vec[num_col[j*dim+i]];
      }
      res[i]=tmp;
    }
}

void compare(float* res1, float* res2, int n){
  int fail=0;
  for(int i=0; i<n; i++){
    float a,b;
    if(res1[i]<0)
      a=res1[i]*(-1);
    else 
      a=res1[i];
    if(res2[i]<0)
      b=res2[i]*(-1);
    else 
      b=res2[i];
    if((a<0.01)&&(b<0.01)){
      continue;
    }
    if(i<10)
      printf("i=%d %lf %lf\n",i,a,b);
    float diff=(a-b)/(a+0.000001);
    if(diff<0)
      diff=diff*(-1);
    if(diff>0.0005)
      fail++;
  }
  printf("Number of errors: %d\n", fail);
}

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}
//The CSR-format matrix is dimXdim that has n non-zero elements.
int initMatrix(int *row, int *col, float *data, int n, int dim)
{
    int nnzAssigned = 0;
    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix
    double prob = (double)n / ((double)dim * (double)dim);

    // Seed random number generator
    srand48(8675309L);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    bool fillRemaining = false;
    int cnt = 0;
    int interval_size = 0;
    for (int i = 0; i < dim; i++)
    {
        row[i] = nnzAssigned;
        cnt = 0;
        for (int j = 0; j < dim; j++)
        {
            int numEntriesLeft = (dim * dim) - ((i * dim) + j);
            int needToAssign = n - nnzAssigned;
            if (numEntriesLeft <= needToAssign) {
                fillRemaining = true;
            }
            if ((nnzAssigned < n && drand48() <= prob) || fillRemaining)
            {
                // Assign (i,j) a value
                col[nnzAssigned] = j;
                data[nnzAssigned] = 1;
                nnzAssigned++;
                cnt++;
            }
        }
        if(cnt > interval_size)
          interval_size = cnt;
   }
    // Observe the convention to put the number of non zeroes at the end of the
    // row delimiters array
    row[dim] = n;
    assert(nnzAssigned == n);
    return interval_size;
}

int main(){

  int dim=30000;
  int n=dim*dim/100;
  int *row = (int*)malloc(sizeof(int)*(dim+1));
  int *col = (int*)malloc(sizeof(int)*n);
  float *data = (float*)malloc(sizeof(float)*n);

  int max_col = initMatrix(row, col, data, n, dim);
  int *num_col = (int*)malloc(sizeof(int)*dim*max_col);

  float *vec = (float*)malloc(sizeof(float)*dim);
  for(int i=0; i<dim; i++){
    vec[i]=1;
  }

  float *val =(float*)malloc(sizeof(float)*dim*max_col);
  for(int i=0; i<dim; i++){
      int num=row[i+1]-row[i];
      for(int j=0; j<num; j++){
          val[j*dim+i]=data[row[i]+j];
          num_col[j*dim+i]=col[row[i]+j];
      }
  }

  float *result = (float*)malloc(sizeof(float)*dim);
  float *result_gpu_res = (float*)malloc(sizeof(float)*dim);

  for(int i=0; i<dim; i++){
    float t = 0;
    for(int j=row[i]; j<row[i+1]; j++){
      int colNum = col[j];
      t += data[j] * vec[colNum];
    }
    result[i] = t;
  }

  int *row_gpu;
  int *col_gpu;
  float *data_gpu;
  float *vec_gpu;
  float *result_gpu;

  int *num_col_gpu;
  float *val_gpu;

  cudaMalloc( (void **)&row_gpu, sizeof(int)*(dim+1));
  cudaMalloc( (void **)&col_gpu, sizeof(int)*n);
  cudaMalloc( (void **)&data_gpu, sizeof(float)*n);
  cudaMalloc( (void **)&vec_gpu, sizeof(float)*dim);
  cudaMalloc( (void **)&result_gpu, sizeof(float)*dim);

  cudaMalloc( (void **)&num_col_gpu, sizeof(int)*(dim*max_col));
  cudaMalloc( (void **)&val_gpu, sizeof(float)*(dim*max_col));

  cudaMemcpy(row_gpu, row, sizeof(int)*(dim+1), cudaMemcpyHostToDevice);
  cudaMemcpy(col_gpu, col, sizeof(int)*n, cudaMemcpyHostToDevice);
  cudaMemcpy(data_gpu, data, sizeof(float)*n, cudaMemcpyHostToDevice);
  cudaMemcpy(vec_gpu, vec, sizeof(float)*dim, cudaMemcpyHostToDevice);

  cudaMemcpy(num_col_gpu, num_col, sizeof(int)*(dim*max_col), cudaMemcpyHostToDevice);
  cudaMemcpy(val_gpu, val, sizeof(float)*(dim*max_col), cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)ceil((float)dim/ ((float)DIM_THREAD_BLOCK_X)), 1);
  
  spmv<<<grid,block>>>(num_col_gpu, val_gpu, vec_gpu, result_gpu, dim, max_col);
  
  cudaThreadSynchronize();
  cudaMemcpy(result_gpu_res, result_gpu, sizeof(float)*dim, cudaMemcpyDeviceToHost);
  compare(result, result_gpu_res, dim);

  
  double time1=timestamp();
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){
    spmv<<<grid,block>>>(num_col_gpu, val_gpu, vec_gpu, result_gpu, dim, max_col);

  }
  cudaThreadSynchronize();
  double time2=timestamp();

  double time = (time2-time1)/ITERATIONS;
  double flops = 2 * (double)n;
  double gflopsPerSecond = flops/(1000000000)/time;
  double dataCopy = sizeof(int)*dim + sizeof(int)*n + sizeof(float)*n + sizeof(float)*dim*2;
  double bandwidth = dataCopy/time/1000000000;
  printf("GFLOPS/s=%lf\n",gflopsPerSecond );
  printf("GB/s=%lf\n",bandwidth );
  printf("GB=%lf\n",dataCopy/1000000000);
  printf("GFLOPS=%lf\n",flops/(1000000000));
  printf("time(s)=%lf\n",time);
  return 0;
}
