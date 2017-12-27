/*-----读者优先------*/
#include <stdio.h>  
#include <stdlib.h>  
#include <sys/types.h>
#include <unistd.h>  
#include <semaphore.h>  
#include <pthread.h>  
#define reader_cost       1               //读者速度  
#define writer_cost       2               //写者速度  

sem_t w_sem, r_sem;  
int readcount=0;        //计算读者数量

void writing(void)                          
{  
    sleep(writer_cost);  
}  
void reading()                                        
{  
    sleep(reader_cost);  
} 
   
void *writer(void *arg)          
{  
    while(1)  {  
        sem_wait(&w_sem);

        printf("writer in this buffer\n"); 
        writing();     

        sem_post(&w_sem); 
        pthread_exit((void *)0);//结束该线程    
    }  
}  
   
void *reader(void *arg) 
{  
    while(1)  {  
        sem_wait(&r_sem); 
        if(readcount == 0)sem_post(&w_sem);  //没有读时方可写
        readcount++; 
        sem_post(&r_sem); 

            printf("redaer in this buffer\n");  
            reading();
        sem_wait(&r_sem);
        --readcount;
        if(readcount == 0)sem_post(&w_sem);  //没有读时方可写
        sem_post(&r_sem);  
        
        pthread_exit((void *)0);//结束该线程             
    }  
}  
   
int main()  
{  
    pthread_t tid_w[10],tid_r[10];  
    int j;
    sem_init(&w_sem,0,1);  
    sem_init(&r_sem,0,5);  

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
    sem_destroy(&r_sem);
    sem_destroy(&w_sem);

    exit(0); 
}  