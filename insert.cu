#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>

__global__ void Insert_Elem(int *heap,int *d_elements,int *curSize,int *lockArr){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int childInd = atomicInc((unsigned *) curSize,11010);
    lockArr[childInd] = 1; //lock child index
   
    heap[childInd] = d_elements[tid];
    
    int parInd = (childInd-1)/2;
    int oldval = 1;
    do
    {
        oldval = atomicCAS(& lockArr[parInd],0,1);

        if(oldval == 0 && heap[parInd] > heap[childInd])
        {
            printf("ThreadId = %d, childId = %d, parInd = %d, childval = %d, parVal = %d\n",tid,childInd,parInd,heap[childInd],heap[parInd]);
            int temp = heap[parInd];    //swapping the elements
            heap[parInd] = heap[childInd];
            heap[childInd] = temp;
            lockArr[childInd] = 0; //unlock the child
            childInd = parInd;
            parInd = (childInd-1)/2;
            oldval = 1;

            //if we have reached the root
            if(childInd == 0){
                printf("End of threadId = %d\n",tid);
                oldval = 0;
                lockArr[childInd] = 0;
            }   
        }
        else if(oldval == 0)
        {
            printf("End of threadId = %d\n",tid);
            lockArr[childInd] = 0;
            lockArr[parInd] = 0;
        }
    }while(oldval  != 0);
    
}


int main() {
    int *d_a;
    int maxSize = 100; 
    int *curSize;
    int *lockArr;

    cudaHostAlloc(&curSize, sizeof(int), 0);

    int h_a[maxSize] = {15,17,18,24,29,35,39};
    *curSize = 7;

    cudaMalloc(&d_a,maxSize*sizeof(int)); 
    cudaMemcpy(d_a,h_a,maxSize * sizeof(int),cudaMemcpyHostToDevice);

    int elemSize = 8;
    int elements[elemSize] = {19,14,1,34,12,89,2,30};

    int *d_elements;
    cudaMalloc(&d_elements,elemSize*sizeof(int));
    cudaMemcpy(d_elements,elements,elemSize * sizeof(int),cudaMemcpyHostToDevice);
    cudaMalloc(&lockArr,(elemSize + *curSize)*sizeof(int));
    cudaMemset(lockArr,0,(elemSize + *curSize)*sizeof(int));

    Insert_Elem<<<elemSize,1>>>(d_a,d_elements,curSize,lockArr);
    // Insert_Elem<<<1,elemSize>>>(d_a,d_elements,curSize,lockArr);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_a,d_a,maxSize*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i = 0;i<*curSize;i++)
        printf("%d, ",h_a[i]);
    return 0;
}



// class Heap{
// 	public:
// 	int size;
//     int maxSize;
// 	int *heap;
//     //also locks will come here
// }

// __device__ void Heap::initHeap(int size)
// {
//     size = 0;
//     heap = 
// }

