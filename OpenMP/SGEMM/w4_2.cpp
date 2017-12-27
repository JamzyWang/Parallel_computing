/*-----OpenMP------------
-------矩阵分块相乘-----*/

#include <iostream>  
#include <omp.h> // OpenMP编程需要包含的头文件
#include <time.h>
#include <stdlib.h>

using namespace std;

#define Max 2048

double A[Max] [Max] = {0};  
double B[Max] [Max] = {0};
double C[Max] [Max] = {0};

void matrixInit()
{
    //#pragma omp parallel for num_threads(64)
    for(int row = 0 ; row < Max ; row++ ) {
        for(int col = 0 ; col < Max ;col++){
            srand(row+col);
            A[row] [col] = rand() % 1000 ;
            B[row] [col] = rand() % 1000 ;
        }
    }
    //#pragma omp barrier
}

void smallMatrixMult (int upperOfRow , int bottomOfRow , 
                      int leftOfCol , int rightOfCol ,
                      int transLeft ,int transRight )
{
    int row=upperOfRow;
    int col=leftOfCol;
    int trans=transLeft;

    #pragma omp parallel for num_threads(64)
    for(int row = upperOfRow ; row <= bottomOfRow ; row++){
        for(int col = leftOfCol ; col < rightOfCol ; col++){
            for(int trans = transLeft ; trans <= transRight ; trans++){
                C[row] [col] += A[row] [trans] * B[trans] [col] ;
            }
        }
    }
    //#pragma omp barrier
}

void matrixMulti(int upperOfRow , int bottomOfRow , 
                 int leftOfCol , int rightOfCol ,
                 int transLeft ,int transRight )
{
    if ( ( bottomOfRow - upperOfRow ) < 512 ) 
        smallMatrixMult ( upperOfRow , bottomOfRow , 
                          leftOfCol , rightOfCol , 
                          transLeft , transRight );

    else
    {
        #pragma omp task
        {
            matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,
                         leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,
                         transLeft , ( transLeft + transRight ) / 2 );
            matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,
                         leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,
                         ( transLeft + transRight ) / 2 + 1 , transRight );
        }

        #pragma omp task
        {
            matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,
                         ( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,
                         transLeft , ( transLeft + transRight ) / 2 );
            matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,
                         ( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,
                         ( transLeft + transRight ) / 2 + 1 , transRight );
        }
        

        #pragma omp task
        {
            matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,
                         leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,
                         transLeft , ( transLeft + transRight ) / 2 );
            matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,
                         leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,
                         ( transLeft + transRight ) / 2 + 1 , transRight );
        }

        #pragma omp task
        {
            matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,
                         ( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,
                         transLeft , ( transLeft + transRight ) / 2 );
            matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,
                         ( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,
                         ( transLeft + transRight ) / 2 + 1 , transRight );
        }

        #pragma omp taskwait
    }
}

int main()  
{ 
    float time_use = 0;
    struct timeval start;
    struct timeval end;

    matrixInit();

    gettimeofday(&start, NULL);

    //smallMatrixMult( 0 , Max - 1 , 0 , Max -1 , 0 , Max -1 );
    matrixMulti( 0 , Max - 1 , 0 , Max -1 , 0 , Max -1 );
    
    gettimeofday(&end, NULL);
    time_use = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    cout << "time_use is "<< time_use/1000000 << endl;
    
    return 0; 
}