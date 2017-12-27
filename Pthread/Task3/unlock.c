/*-----Concurrent Linked List----*/
/*-----CAS/SkipList----*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

typedef struct _node_t { // basic node structure
  int     key;
  struct _node_t *next;
}node_t;

typedef struct mypara {
  struct _node_t *old;
  struct _node_t *_new;
}mypara;

void* insert(void* arg) {
	// synchronization not needed
	node_t *prev = ((mypara*)arg)->old;
	node_t *node = ((mypara*)arg)->_new;
    node_t *next = malloc(sizeof(node_t));
    //printf("prev.key=%d\n",((mypara*)arg)->old->key);
	while (1) {
	    next = prev->next;
       // printf("next.key=%d\n",next->key);
		while (next != NULL && next->key < node->key) {
			prev = next;
			next = prev->next;
		}
		node->next = next;
		if (__sync_bool_compare_and_swap(&prev->next, node->next, node))break;
	}
}

void* _delete(void* arg) {

	node_t *prev = ((mypara*)arg)->old;

	if(prev->next == NULL){
		printf("Empty\n");
		exit(0);
	}
	node_t *node = ((mypara*)arg)->_new;

    printf("prev->key = %d\nnew->key = %d\n", prev->key,node->key);
    
	while(1) {
        //break;
		while((prev !=NULL)&&(prev->key != node->key)){
            prev = prev->next;
            printf("_prev->key = %d\n", prev->key);
		}//找到目标之后做删除操作
        if(prev == NULL){
            //printf("no exists\n");
            break;
        }
		if(__sync_bool_compare_and_swap(&prev->next, node, node->next)){
			//如果prev->next还是指向node，就将其改为node->next
			printf("node->key = %d\n", node->key);
			break;
		}
        

	}
}
void insert_output(node_t *head){
    while(head->next != NULL){
     //   printf("node.key =%d \n",head->key);
        head = head->next;
        break;
    }
}
int main()
{
    float time_use = 0;
    struct timeval start;
    struct timeval end;

    gettimeofday(&start, NULL);

    pthread_t tpid;
    //int p_n=10, i, key[10]={1,6,3,8,34,29,25,47,38,85};
    int p_n=3, i, key[15]={1,6,3,8,34,29,25,47,38,85,1,6,3,8,34};
    //scanf("%d",&p_n);
    //thread_handles = (pthread_t*)malloc(p_n * sizeof(pthread_t));
    node_t *head;
    head = malloc(sizeof(node_t));
    head->next = NULL;
    head->key = 0;

	node_t *pre;
    mypara *tmp;
    tmp = malloc(sizeof(mypara));
    tmp->old = head;
 
    for(i=0; i< p_n; i++){
        //scanf("%d",&key);
       // printf("key = %d\n",key);
    	pre = malloc(sizeof(node_t));
        pre->key = key[i];
        pre->next = NULL;
     	tmp->_new = pre;
       // printf("tmp.new.key=%d\n",tmp->_new->key);

        pthread_create(&tpid,NULL,insert,(void*)tmp);
    }
    for(i=0; i< p_n; i++){
      pthread_join(tpid,NULL);
    }

  /* -------------分割线-------*/
    printf("test Insert end\n");
     pre = malloc(sizeof(node_t));
    pre->key = 3;
    pre->next = NULL;
    tmp->_new = pre;
    //删除key=3 的那一个
   for(i=0; i< p_n; i++){
        pthread_create(&tpid,NULL,_delete,(void*)tmp);
    }
    for(i=0; i< p_n; i++){
      pthread_join(tpid,NULL);
    }
    gettimeofday(&end, NULL);
    time_use = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);

  //  insert_output(head);
    printf("time_use is %f\n", time_use/1000000);
    //insert_output(head);
    return 0;
}

