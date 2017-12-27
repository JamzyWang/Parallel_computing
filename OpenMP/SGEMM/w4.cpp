/*---OpenMP----*/
/*---在终端下需要vim？----*/
#include <iostream>  
#include <omp.h> // OpenMP编程需要包含的头文件
#include <sys/time.h>
#include <stdlib.h>

using namespace std;

#define Max 256

int A[Max][Max] = {0};  
int B[Max][Max] = {0};
int C[Max][Max] = {0};

void matrix_Init()
{
    #pragma omp parallel for num_threads(64)
    for(int row = 0 ; row < Max ; row++ ) {
        for(int col = 0 ; col < Max ;col++){
            srand(row+col);
            A[row][col] = rand() % 1000;    //0--1000之间的随机整数
            B[row][col] = rand() % 1000;
        }
    }
    //#pragma omp barrier
}

/* -----------矩阵相乘------------*/
//计算C[row][col]
/*int calcu_part_multi(int row,int col)
{
    int res = 0;
    for(int t = 0 ; t < Max ; t++) {
        res += A[row][t] * B[t][col] ;
    }
    return res;
}*/

void matrix_Multi()
{
    #pragma omp parallel for num_threads(64)

    #pragma omp parallel shared(A,B,C)private(i,j,k)
    {
        #pragma omp for schedule(dynamic)
        for(int i = 0 ; i < Max ; i++){
            for(int j = 0; j < Max ; j++){
                for(int k =0; k < Max; k++)
                    C[i][j] += A[i][k]*B[k][j];
            }
        }
    }    
    //#pragma omp barrier
}

int main()  
{ 
    float time_use = 0;
    struct timeval start;
    struct timeval end;
    matrix_Init();

    gettimeofday(&start, NULL);

    matrix_Multi();
    
    gettimeofday(&end, NULL);
    time_use = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    cout << "time_use is "<< time_use/1000000 << endl;
    return 0;  
}