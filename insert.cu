#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include <time.h>

__global__ void Insert_Elem(volatile int *heap,volatile int *d_elements,int *curSize,volatile int *lockArr,int *elemSize){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < *elemSize)
    {
        int childInd = atomicInc((unsigned *) curSize,11010);
        // lockArr[childInd] = 1; //lock child index
        
        // __syncthreads();
        heap[childInd] = d_elements[tid];
        // printf("%d,",heap[childInd]);
        
        int parInd = (childInd-1)/2;
        int oldval = 1;
        do
        {
            oldval = atomicCAS((int*)&lockArr[parInd],0,1);
            if(oldval == 0)
            {
                if(heap[parInd] >= heap[childInd])
                {
                    // printf("ThreadId = %d, childId = %d, parInd = %d, childval = %d, parVal = %d\n",tid,childInd,parInd,heap[childInd],heap[parInd]);
                    int temp = heap[parInd];    //swapping the elements
                    heap[parInd] = heap[childInd];
                    heap[childInd] = temp;
                    lockArr[childInd] = 0; //unlock the child
                    __threadfence();
                    // printf("x");
                    // printf("value after swap ThreadId = %d, childId = %d, parInd = %d, childval = %d, parVal = %d\n",tid,childInd,parInd,heap[childInd],heap[parInd]);
                    childInd = parInd;
                    parInd = (childInd-1)/2;
                    

                    //if we have reached the root
                    if(childInd == 0){
                        // printf("End of threadId = %d\n",tid);
                        oldval = 0;
                        lockArr[childInd] = 0;
                    } 
                    else
                    {
                        oldval = 1;
                    } 
                }
                else
                {
                    // printf("End of threadId = %d\n",tid);
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
    // printf("\nTotal Size is %d",size);
    for(int i = 0;i<size/2;i++)
    {
        if(ar[i] > ar[2*i + 1]) return false;
        if((2*i + 2) < size && ar[i] > ar[2*i + 2]) return false;
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
        elements[i] = getRandom(1,5000);
    }
}

__global__ void setLockVar(int *lockArr)
{
    for(int i = 1;i<1028;i++)
        lockArr[i] = 1;
}

int main() {
    srand(time(0));
    int *d_a;
    int maxSize = 1028; 
    int *curSize;
    int *lockArr;
    int *elemSize;
    int countvalid = 0;
    for(int lk = 0;lk<1000;lk++)
    {
        cudaHostAlloc(&curSize, sizeof(int), 0);
        cudaHostAlloc(&elemSize, sizeof(int), 0);

        int h_a[maxSize] = {15};
        *curSize = 1;

        cudaMalloc(&d_a,maxSize*sizeof(int)); 
        cudaMemcpy(d_a,h_a,maxSize * sizeof(int),cudaMemcpyHostToDevice);

        *elemSize = getRandom(1,1020);
        int elements[*elemSize];
        // int elements[elemSize] = {19,14,1,34,12,89,2,30};
        FillArray(elements,*elemSize);
        printf("Inserted Elements are = %d\n",*elemSize);
        // printArray(elements,elemSize);

        int *d_elements;
        cudaMalloc(&d_elements,*elemSize*sizeof(int));
        cudaMemcpy(d_elements,elements,*elemSize * sizeof(int),cudaMemcpyHostToDevice);
        cudaMalloc(&lockArr,(*elemSize + *curSize)*sizeof(int));
        cudaMemset(lockArr,0,(*elemSize + *curSize)*sizeof(int));
        setLockVar<<<1,1>>>(lockArr);
        cudaDeviceSynchronize();
        // int block = ceil((float) *elemSize/1024);

        // Insert_Elem<<<block,1024>>>(d_a,d_elements,curSize,lockArr,elemSize);
        Insert_Elem<<<*elemSize,1>>>(d_a,d_elements,curSize,lockArr,elemSize);
        // Insert_Elem<<<1,*elemSize>>>(d_a,d_elements,curSize,lockArr,elemSize);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_a,d_a,maxSize*sizeof(int),cudaMemcpyDeviceToHost);
        // printf("\nHeap is :");
        // printArray(h_a,*curSize);
        
        if(checkHeap(h_a,*curSize)) countvalid++;
    }
    cudaDeviceSynchronize();
    printf("\nvalid : %d",countvalid);
    return 0;
}
