#include <iostream>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <cassert>

#define ITERATIONS 10
using namespace std;

#define thread_num 10
#define dim 500

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}
//The CSR-format matrix is dimXdim that has n non-zero elements.
void initMatrix(int *row, int *col, float *data, int n){
    int nnzAssigned = 0;
    // Figure out the probability that a nonzero should be assigned to a given
    // spot in the matrix

    double prob = (double)n / ((double)dim * (double)dim);

    
    // Seed random number generator
    srand48(8675309L);

    // Randomly decide whether entry i,j gets a value, but ensure n values
    // are assigned
    bool fillRemaining = false;
  //  #pragma omp parallel for num_threads(10)
    for (unsigned short i = 0; i < dim; i++)
    {
        row[i] = nnzAssigned;
        for (unsigned short j = 0; j < dim; j++)
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
            }
        }
    }
    // Observe the convention to put the number of non zeroes at the end of the
    // row delimiters array
    row[dim] = n;
    assert(nnzAssigned == n);
}

int main()
{
  int n = dim*dim/100;
  int *row = (int*)malloc(sizeof(int)*dim);
  int *col = (int*)malloc(sizeof(int)*n);
  int *tmp = (int*)malloc(sizeof(int)*dim);

  float *data = (float*)malloc(sizeof(float)*n);
  float *vec = (float*)malloc(sizeof(float)*dim);
  float *result = (float*)malloc(sizeof(float)*dim);


  initMatrix(row, col, data, n);

  for(int i=0; i<dim; i++){
    vec[i]=1;
  }
  for(int i=1; i<=dim; i++)
      tmp[i-1] = row[i];
   // 循环分离，减少依赖

  double time1=timestamp();

    for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){
      #pragma omp parallel for num_threads(thread_num) schedule(guided)
      for(unsigned short i=0; i<dim; i++){
        float t = 0;
        for(int j=row[i]; j<tmp[i]; j++){
          unsigned short colNum = col[j];
          t += data[j] * vec[colNum];
        }
        result[i] = t;
      }
    }

  
  double time2=timestamp();

  double gflop = 2 * (double)n;

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