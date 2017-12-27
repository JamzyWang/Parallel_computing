#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

#define N 1024
#define ITERATIONS 10
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32
#define WIDTH_SIZE 32
using namespace std;

/*--------GPU跑的函数--------*/
//sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, a, b);
__global__ void sgemm(float *A, float *B, float *C, int n, float a, float b) {
    __shared__ float sharedM[WIDTH_SIZE][WIDTH_SIZE];
    __shared__ float sharedN[WIDTH_SIZE][WIDTH_SIZE];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by*WIDTH_SIZE + ty;
    int col = bx*WIDTH_SIZE + tx;
    float v = 0;

    for (int i = 0; i < (int)(ceil((float)N/WIDTH_SIZE)); ++i){
        sharedM[ty][tx] = A[row*N + i*WIDTH_SIZE + tx];
        sharedN[ty][tx] = B[(i*WIDTH_SIZE + ty)*N + col];
        __syncthreads();
        //有一个if-else优化的办法是使用cudaMallocPitch
        //在分配的时候就自动设定边界,但是测试效果不怎么样
        //还是跟内存大小有关，如果分配的是32整数倍，不需要if-else

        #pragma unroll    //提升了10GFLOPS，效果明显
        for(int j = 0; j < WIDTH_SIZE; ++j){
            v += sharedM[ty][j] * sharedN[j][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N)
      C[row*N+col] = a*v +b*C[row*N+col];
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
    //if(i<10)
    //  printf("i=%d %lf %lf\n",i,a,b);
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

int main(){
  float A[N*N], B[N*N],C_cpu[N*N], C_gpu_final[N*N];//
  //float A[N][N], B[N][N], C_cpu[N][N], C_gpu_final[N][N];
  float a=0.5, b=0.3;

  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      A[i*N+j]=(float)rand()/(float)(RAND_MAX/a);
      B[i*N+j]=(float)rand()/(float)(RAND_MAX/a);
      C_cpu[i*N+j]=0;
      C_gpu_final[i*N+j]=0;
    }
  }

  for(int j=0; j<N; j++){
    for(int i=0; i<N; i++){
      C_cpu[i*N+j]+=b*C_cpu[i*N+j];
      for(int k=0; k<N; k++){
        C_cpu[i*N+j] += a*A[i*N+k]*B[k*N+j];
      }
    }
  }

  float *A_gpu;
  float *B_gpu;
  float *C_gpu;
  cudaMalloc((void **)&A_gpu, sizeof(float)*N*N);
  cudaMalloc((void **)&B_gpu, sizeof(float)*N*N);
  cudaMalloc((void **)&C_gpu, sizeof(float)*N*N);

  cudaMemcpy(A_gpu, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C_gpu_final, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)ceil( ((float)N) / ((float)block.x) ), (size_t)ceil( ((float)N) / ((float)block.y)) );
  //取整函数ceil
  sgemm<<<grid, block>>>(A_gpu, B_gpu, C_gpu, N, a, b);
  cudaThreadSynchronize();

  cudaMemcpy(C_gpu_final, C_gpu, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
  compare(C_cpu, C_gpu_final, N*N);   
  //用于比较CPU和GPU上的计算误差，在double的时候可能会有问题

  /*----Cuda优化由此开始------*/
  double time1=timestamp();
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){
    sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, a, b);
    cudaThreadSynchronize();

  }
  double time2=timestamp();

  double time = (time2-time1)/ITERATIONS;
  double flops = 2*N*N*N;
  double gflopsPerSecond = flops/(1000000000)/time;
  double GB = (double)(N)*N*4/1000000000;
  double GBpS = (double)(N)*N*4/1000000000/time;
  printf("GFLOPS/s=%lf\n",gflopsPerSecond );
  printf("GB/s=%lf\n",GBpS);
  printf("GFLOPS=%lf\n",flops/(1000000000));
  printf("GB=%lf\n",GB);
  printf("time(s)=%lf\n",time);

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  return 0;
}
