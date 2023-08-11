#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include <time.h>
#define maxSize 200000

__global__ void Insert_Elem(volatile int *heap,int *d_elements,int *curSize,volatile int *lockArr,int *elemSize){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < *elemSize)
    {
        int childInd = atomicInc((unsigned *) curSize,200010);
        heap[childInd] = d_elements[tid];
        // __threadfence();

        int parInd = (childInd-1)/2;
        int oldval = 1;
        do
        {
            oldval = atomicCAS((int*)&lockArr[parInd],0,1);
            if(oldval == 0)
            {
                if(heap[parInd] > heap[childInd])
                {
                    // printf("ThreadId = %d, childId = %d, parInd = %d, childval = %d, parVal = %d\n",tid,childInd,parInd,heap[childInd],heap[parInd]);
                    int temp = heap[parInd];    //swapping the elements
                    heap[parInd] = heap[childInd];
                    heap[childInd] = temp;

                    __threadfence();

                    lockArr[childInd] = 0; //unlock the child
    
                    childInd = parInd;
                    parInd = (childInd-1)/2;
                    oldval = 1;

                    //if we have reached the root
                    if(childInd == 0){
                        oldval = 0;
                        lockArr[childInd] = 0;
                    }  
                }
                else
                {
                    lockArr[childInd] = 0;
                    lockArr[parInd] = 0;
                } 
                
            }
            __threadfence();
        }while(oldval  != 0);
    }
}
bool checkHeap(int *ar,int size)
{
    for(int i = 0;i<size/2;i++)
    {
        if(ar[i] > ar[2*i + 1]){
            printf("\nproblem found at index parent = %d,child = %d\n",i,2*i + 1);
            printf("\nproblem found at index parentval = %d,childval = %d\n",ar[i],ar[2*i + 1]); 
            return false;
        } 
        if((2*i + 2) < size && ar[i] > ar[2*i + 2]){
            printf("\nproblem found at index parent = %d,child = %d\n",i,2*i + 2);
            printf("\nproblem found at index parentval = %d,childval = %d\n",ar[i],ar[2*i + 2]);
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

int main() {
    srand(time(0));
    int countvalid = 0;
    int inivalid = 0;
    
    for(int lk = 0;lk<1000;lk++)
    {
        int *d_a;
        int *curSize;
        int *lockArr;
        int *elemSize;
        cudaHostAlloc(&curSize, sizeof(int), 0);
        cudaHostAlloc(&elemSize, sizeof(int), 0);

        int h_a[maxSize];
        int num = pow(2,getRandom(1,16))-1;
        // int num = pow(2,getRandom(1,16);
        *curSize = num;
        FillArray(h_a,*curSize);
        // printf("Initial random elements is : ");
        // printArray(h_a,*curSize);
        buildHeap(h_a,*curSize);
        // printf("\nAfter Heapify :");
        // printArray(h_a,*curSize);
        if(checkHeap(h_a,*curSize)) inivalid++;
        
        cudaMalloc(&d_a,maxSize*sizeof(int)); 
        cudaMemcpy(d_a,h_a,maxSize * sizeof(int),cudaMemcpyHostToDevice);

        *elemSize = getRandom(1,num);
        int elements[*elemSize];
        // int elements[elemSize] = {19,14,1,34,12,89,2,30};
        FillArray(elements,*elemSize);
        // printf("%d. No of Inserted Elements are = %d , ",inivalid,*elemSize);
        // printf("%d,",inivalid);
        // printArray(elements,*elemSize);

        int *d_elements;
        cudaMalloc(&d_elements,*elemSize*sizeof(int));
        cudaMemcpy(d_elements,elements,*elemSize * sizeof(int),cudaMemcpyHostToDevice);
        cudaMalloc(&lockArr,(*elemSize + *curSize)*sizeof(int));
        cudaMemset(lockArr,0,(*elemSize + *curSize)*sizeof(int));
        // setLockVar<<<1,1>>>(curSize,lockArr);
        cudaDeviceSynchronize();
       
        int block = ceil((float) *elemSize/1024);
        Insert_Elem<<<block,1024>>>(d_a,d_elements,curSize,lockArr,elemSize);
        // Insert_Elem<<<*elemSize,1>>>(d_a,d_elements,curSize,lockArr,elemSize);
        // Insert_Elem<<<1,*elemSize>>>(d_a,d_elements,curSize,lockArr,elemSize);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_a,d_a,maxSize*sizeof(int),cudaMemcpyDeviceToHost);
        // printf("\nHeap is :");
        // printArray(h_a,*curSize);
        // printf("\n\n\n");
        
        if(checkHeap(h_a,*curSize)) {
            // printf("Valid\n");
            countvalid++;
        }
    }
    printf("\nIni valid : %d",inivalid);
    printf("\nvalid : %d\n\n",countvalid);
    return 0;
}
