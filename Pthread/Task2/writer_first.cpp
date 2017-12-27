/*-----写者优先------*/
#include <stdio.h>  
#include <stdlib.h>  
#include <sys/types.h>
#include <unistd.h>  
#include <semaphore.h>  
#include <pthread.h>  

#define reader_cost       2               //读者速度  
#define writer_cost       1               //写者速度  

sem_t w_sem, r_sem, x_sem, y_sem, z_sem;  
int readcnt = 0,writecnt = 0;  
   
void writing(void)                          
{  
    sleep(writer_cost);  
}  
void reading()                                       
{  
    sleep(writer_cost);  
} 

void *writer(void *arg)          
{  
    while(1)  {  
        sem_wait(&y_sem); //互斥对writecnt操作
            writecnt++;
            if(writecnt == 1)sem_wait(&r_sem);//查看是否有读           
        sem_post(&y_sem);

        sem_wait(&w_sem);//只能有一个写
        printf("writer is in this buffer\n"); 
        writing();    
        sem_post(&w_sem);    

        sem_wait(&y_sem);  
            --writecnt;
            if(writecnt == 0) sem_post(&r_sem);
        sem_post(&y_sem);
        pthread_exit((void *)0);//结束该线程
    }  
}  
   
void *reader(void *arg) //读者线程  
{  
    while(1)  {  
        sem_wait(&z_sem);           //用于存放多个线程 
            sem_wait(&r_sem);       //读线程的等待栈
                sem_wait(&x_sem);   //专门为READCNT设置
                    readcnt++;      //判断是否有读进程
                    if(readcnt == 1)sem_wait(&w_sem);
                    printf("redaer is in this buffer\n");  
                sem_post(&x_sem);        
            sem_post(&r_sem);       //读者信号量加一  
        sem_post(&z_sem); 
        reading();      
        sem_wait(&x_sem);
            --readcnt;
            if(readcnt == 0) sem_post(&w_sem);//无读方可写
        sem_post(&x_sem);   
        pthread_exit((void *)0);//结束该线程   
    }  
}  
   
int main()  
{  
    pthread_t tid_w[10],tid_r[10];  
    int j;
    sem_init(&x_sem,0,1);    
    sem_init(&y_sem,0,1);    
    sem_init(&z_sem,0,1);  
    sem_init(&w_sem,0,5); 
    sem_init(&r_sem,0,1); 

    for(j=0;j<10;j++) {
        pthread_create(tid_w+j,NULL,writer,NULL);  
        printf("writer[%d] is created\n",j);
        pthread_create(tid_r+j,NULL,reader,NULL); 
        printf("reader[%d] is created\n",j); 
    }
   
    for(j=0;j<10;j++) {
        pthread_join(tid_w[j],NULL); 
        printf("writer[%d] is finished\n",j);
    }
    for(j=0;j<10;j++) {
        pthread_join(tid_r[j],NULL);    //回收线程
        printf("reader[%d] is finished\n",j); 
    }
    sem_destroy(&x_sem);    //清除信号量
    sem_destroy(&y_sem);
    sem_destroy(&r_sem);
    sem_destroy(&w_sem);

    exit(0); 
}  