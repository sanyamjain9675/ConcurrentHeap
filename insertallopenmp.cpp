#include <stdio.h>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <omp.h>
#include <time.h>
#include <cstdlib>

//total size of the heap
#define maxSize 100000
omp_lock_t lock[maxSize];

bool checkHeap(int *ar,int size)
{
    for(int i = 0;i<size/2;i++)
    {
        if(ar[i] > ar[2*i + 1]){
            printf("\nproblem found at index parent = %d,child = %d\n",i,2*i + 1);
            printf("problem found at index parentval = %d,childval = %d\n",ar[i],ar[2*i + 1]); 
            return false;
        } 
        if((2*i + 2) < size && ar[i] > ar[2*i + 2]){
            printf("\nproblem found at index parent = %d,child = %d\n",i,2*i + 2);
            printf("problem found at index parentval = %d,childval = %d\n",ar[i],ar[2*i + 2]);
            return false;
        }
    }
    return true;
}
int getRandom(int lower, int upper)
{
    int num = (rand() % (upper - lower + 1)) + lower;
    return num;  
}
void printArray(int arr[],int size)
{
    for(int i = 0;i<size;i++)
        printf("%d, ",arr[i]);
}
void FillArray(int elements[],int size)
{
    for(int i = 0;i<size;i++)
    {
        elements[i] = getRandom(1,1000);
    }
}
    
void heapify(int hp[],int ind,int size)
{
    while(1)
    {
        int leftChild = 2*ind+1;
        int rightChild = 2*ind+2;
        int largeInd = -1;
        if(rightChild < size && hp[ind] > hp[rightChild]){
            if(hp[leftChild] < hp[rightChild])
                largeInd = leftChild;
            else
                largeInd = rightChild;
        }
        else if(leftChild < size && hp[ind] > hp[leftChild]){
            largeInd = leftChild;
        }
        
        if(largeInd == -1)  return;
        int temp = hp[ind];
        hp[ind] = hp[largeInd];
        hp[largeInd] = temp;
        ind = largeInd;
    }

}

void buildHeap(int hp[],int n)
{
    for(int i = n/2 -1 ; i>=0;i--)
    {
        heapify(hp,i,n);
    }
}

double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main() {
    srand(time(0));
    int countvalid = 0;
    int inivalid = 0;

    for(int i = 0;i<maxSize;i++)
        omp_init_lock(&(lock[i]));

    for(int lk = 0;lk<100;lk++)
    {
        int curSize;
        int elemSize;

        int heap[maxSize];
        curSize = getRandom(1,maxSize/10);

        //Initialise Heap with some random values
        FillArray(heap,curSize);

       //heapify the heap
        buildHeap(heap,curSize);

        // printf("Heap before Insertion : ");
        // printArray(heap,curSize);

       //check if satisfies the heap property
        if(checkHeap(heap,curSize)) inivalid++;

        elemSize = getRandom(1,maxSize-curSize-2);
        int elements[elemSize];
        
        FillArray(elements,elemSize);
        printf("\n%d. No of Inserted Elements are = %d\n",inivalid,elemSize);

        double starttime = rtclock(); 

        /* Kernel starts*/
        int count = 0;int ind,childInd;

        #pragma omp parallel for private(ind,childInd)
        for(int i = 0;i<elemSize;i++)
        {
            #pragma omp critical
            {
                ind = count++;
                childInd = curSize++;
            }

            omp_set_lock(&(lock[childInd]));            
            heap[childInd] = elements[count];   

            int parInd = (childInd-1)/2;
            while (1)
            {   
                omp_set_lock(&(lock[parInd]));
                if(heap[parInd] > heap[childInd])
                {
                    int temp = heap[parInd];    //swapping the elements
                    heap[parInd] = heap[childInd];
                    heap[childInd] = temp;

                    omp_unset_lock(&(lock[childInd]));
                    childInd = parInd;
                    parInd = (childInd-1)/2;

                    if(childInd == 0){
                        omp_unset_lock(&(lock[childInd]));
                        break;
                    }  
                }
                else{
                    omp_unset_lock(&(lock[childInd]));
                    omp_unset_lock(&(lock[parInd]));
                    break;
                }
            }
        }
    
        /* Kernel ends*/

        double endtime = rtclock();  

        printtime("GPU Kernel time: ", starttime, endtime);
        
        if(checkHeap(heap,curSize)) {
            countvalid++;
        }
        // printf("Heap after Insertion : ");
        // printArray(heap,curSize);
    }

    for(int i = 0;i<maxSize;i++)
            omp_destroy_lock(&(lock[i]));

    printf("\nInitial valid : %d",inivalid);
    printf("\nvalid : %d\n\n",countvalid);
    return 0;
}