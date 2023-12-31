#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include <time.h>

//total size of the heap
#define maxSize 1000

__global__ void Insert_Elem(volatile int *heap,int *d_elements,int *curSize,volatile int *lockArr,int *elemSize){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < *elemSize)
    {
        int childInd = atomicInc((unsigned *) curSize,maxSize+10);
        heap[childInd] = d_elements[tid];

        int parInd = (childInd-1)/2;

        if(childInd == 0){
            lockArr[childInd] = 0;
        }

        if(childInd != 0)
        {
            int oldval = 1;
            do
            {
                oldval = atomicCAS((int*)&lockArr[parInd],0,1);
                if(oldval == 0) //if we got the lock on parent
                {
                    if(heap[parInd] > heap[childInd])
                    {
                        int temp = heap[parInd];    //swapping the elements
                        heap[parInd] = heap[childInd];
                        heap[childInd] = temp;

                        __threadfence();//necessary

                        lockArr[childInd] = 0; //unlock the child
        
                        childInd = parInd;
                        parInd = (childInd-1)/2;
                        oldval = 1; //we need to heapify again

                        //if we have reached the root
                        if(childInd == 0){
                            oldval = 0; //we need not heapify again
                            lockArr[childInd] = 0;
                        }  
                    }
                    else //if heap property satisfied release the locks
                    {
                        lockArr[childInd] = 0;
                        lockArr[parInd] = 0;
                    } 
                    
                }
                // __threadfence(); //doesnt seem necessary
            }while(oldval != 0);
        }
    }
}
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

__global__ void setLockVar(int *curSize,int *lockArr,int *elemSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < *elemSize)
        lockArr[tid + *curSize] = 1;
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
    
    for(int lk = 0;lk<100;lk++)
    {
        int *d_a;
        int *curSize;
        int *lockArr;
        int *elemSize;
        cudaHostAlloc(&curSize, sizeof(int), 0);
        cudaHostAlloc(&elemSize, sizeof(int), 0);

        int h_a[maxSize];
        // *curSize = getRandom(1,maxSize/10);
        *curSize = 0;

        //Initialise Heap with some random values
        FillArray(h_a,*curSize);

       //heapify the heap
        buildHeap(h_a,*curSize);

       //check if satisfies the heap property
        if(checkHeap(h_a,*curSize)) inivalid++;

        cudaMalloc(&d_a,maxSize*sizeof(int)); 
        cudaMemcpy(d_a,h_a,maxSize * sizeof(int),cudaMemcpyHostToDevice);

        *elemSize = getRandom(1,maxSize-*curSize-2);
        int elements[*elemSize];
        
        FillArray(elements,*elemSize);
        printf("%d. No of Inserted Elements are = %d\n",inivalid,*elemSize);

        int *d_elements;
        cudaMalloc(&d_elements,*elemSize*sizeof(int));
        cudaMemcpy(d_elements,elements,*elemSize * sizeof(int),cudaMemcpyHostToDevice);
        cudaMalloc(&lockArr,(*elemSize + *curSize)*sizeof(int));
        cudaMemset(lockArr,0,(*elemSize + *curSize)*sizeof(int));
    
        int block = ceil((float) *elemSize/1024);

        double starttime = rtclock(); 
        setLockVar<<<block,1024>>>(curSize,lockArr,elemSize);
        cudaDeviceSynchronize();
        Insert_Elem<<<block,1024>>>(d_a,d_elements,curSize,lockArr,elemSize);
        cudaDeviceSynchronize();
        double endtime = rtclock();  
        printtime("GPU Kernel time: ", starttime, endtime);
        cudaMemcpy(h_a,d_a,maxSize*sizeof(int),cudaMemcpyDeviceToHost);
        
        if(checkHeap(h_a,*curSize)) {
            // printf("Valid\n");
            countvalid++;
        }
    }
    printf("\nInitial valid : %d",inivalid);
    printf("\nvalid : %d\n\n",countvalid);
    return 0;
}
