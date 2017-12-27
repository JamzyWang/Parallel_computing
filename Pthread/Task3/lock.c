/*-----Concurrent Linked List----*/
/*-----hand-over-hand-locking----*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
typedef struct _node_t { // basic node structure
  int     key;
  struct _node_t *next;
}node_t;

typedef struct _list_t { // basic list structure
  node_t          *head;
  pthread_mutex_t lock;
}list_t;

typedef struct mypara {
  int k;
  list_t *L;
}mypara;

void List_Init(list_t *L) {
   L->head = NULL;
   pthread_mutex_init(&L->lock, NULL);
}

void* List_Insert(void* arg) {
  // synchronization not needed
  int key = ((mypara*)arg)->k;
  list_t* L = ((mypara*)arg)->L;
  node_t *new = malloc(sizeof(node_t));
  if (new == NULL) {
      perror("malloc");
      exit(0);
  }
  new->key = key;

// just lock critical section
  pthread_mutex_lock(&L->lock);
  new->next = L->head;
  L->head = new;
  pthread_mutex_unlock(&L->lock);
}

void* List_Lookup(void* arg) {
   int key = ((mypara*)arg)->k;
   list_t* L = ((mypara*)arg)->L;
   int rv = -1;
   pthread_mutex_lock(&L->lock);
   node_t *curr = L->head;
   while (curr) {
      if (curr->key == key) {
         rv = 0;
         break;
      }
      curr = curr->next;
   }
   pthread_mutex_unlock(&L->lock);
}


int main()
{
    float time_use = 0;
    struct timeval start;
    struct timeval end;

    gettimeofday(&start, NULL);

    pthread_t tpid;
    //int p_n=10, i, key[10]={1,6,3,8,34,29,25,47,38,85};
    int p_n=15, i, key[15]={1,6,3,8,34,29,25,47,38,85,1,6,3,8,34}; 
    //scanf("%d",&p_n);
    //thread_handles = (pthread_t*)malloc(p_n * sizeof(pthread_t));
    list_t *list;
    list = malloc(sizeof(list_t));
    list->head = NULL;
    pthread_mutex_init(&(list->lock),NULL);

    mypara* tmp;
    tmp = malloc(sizeof(mypara));
    tmp->L = list;
    for(i=0; i< p_n; i++){
        //scanf("%d",&key);
       // printf("key = %d\n",key);
        tmp->k = key[i];
        pthread_create(&tpid,NULL,List_Insert,(void*)tmp);
    }
    for(i=0; i< p_n; i++){
      pthread_join(tpid,NULL);
    }

    
    /*-------------分割线-------*/
  //  printf("test List_Insert end\n");
    tmp->k = 0;
    for(i=0; i< p_n; i++){
        pthread_create(&tpid,NULL,List_Lookup,(void*)tmp);
    }
    for(i=0; i< p_n; i++){
      pthread_join(tpid,NULL);
    }

    gettimeofday(&end, NULL);
    time_use = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);

    printf("time_use is %f\n", time_use/1000000);
    return 0;
}
