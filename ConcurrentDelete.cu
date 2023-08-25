#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include <time.h>
#include <algorithm>

__device__ int globind = 0;

//total size of the heap
#define maxSize 32
#define range 1024

__global__ void init()
{
    globind = 0;
}

__global__ void Del_Elem(volatile int *heap,volatile int *curSize,volatile int *lockArr,volatile int *elemSize,volatile int *newSize,int *d_delElem){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int count = 0;
    int level = 0;
    if(tid < *elemSize)
    {
        int parInd = 0;
        int oldval = 1;
        do
        {
            oldval = atomicCAS((int*)&lockArr[level],0,1);
            if(oldval == 0) //if we got the lock on that level
            {
                if(count == 0)
                {
                    int childInd = atomicDec((unsigned *) curSize,maxSize+10);
                    d_delElem[globind] = heap[parInd];
                    globind++;
                    // printf("Root Value before del = %d,%d, ",heap[parInd],childInd);
                    heap[parInd] = heap[childInd-1];
                    // printf("Root Value after del = %d\n",heap[parInd]);
                    __threadfence();//necessary
                }
                if((level - 2) >= 0) //release the lock before 2 levels
                {
                    lockArr[level-2] = 0;
                }

                int leftChild = 2*parInd + 1;
                int rightChild = 2*parInd + 2;

                int largeInd = -1;

                if(leftChild < *curSize)
                {   
                    if(rightChild < *curSize && heap[parInd] > heap[rightChild]){
                        if(heap[leftChild] < heap[rightChild])
                            largeInd = leftChild;
                        else
                            largeInd = rightChild;
                    }
                    else if(leftChild < *curSize && heap[parInd] > heap[leftChild]){
                        largeInd = leftChild;
                    }
        
                    if(largeInd == -1) //that means no longer need to go down
                    {
                        lockArr[level] = 0; //release current level lock
                        if((level-1) >= 0) lockArr[level-1] = 0; //release previous level lock
                    }
                    else
                    {
                        // printf("value before swap : %d,%d\n",heap[parInd],heap[largeInd]);
                        int temp = heap[largeInd];    //swapping the elements
                        heap[largeInd] = heap[parInd];
                        heap[parInd] = temp;

                        __threadfence();//necessary

                        // printf("value after swap : %d,%d\n",heap[parInd],heap[largeInd]);
        
                        parInd = largeInd;
                        oldval = 1; //we need to heapify again 
                        level = level+1;
                    }
                }
                else //if heap property satisfied release the locks
                {
                    lockArr[level] = 0; //release current level lock
                    if((level-1) >= 0) lockArr[level-1] = 0; //release previous level lock
                } 
                count = count + 1;
            }
            __threadfence(); //doesnt seem necessary
        }while(oldval != 0);
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
        elements[i] = getRandom(1,range);
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
void merge(int array[], int const left, int const mid,
           int const right)
{
    int const subArrayOne = mid - left + 1;
    int const subArrayTwo = right - mid;
 
    // Create temp arrays
    auto *leftArray = new int[subArrayOne],
         *rightArray = new int[subArrayTwo];
 
    // Copy data to temp arrays leftArray[] and rightArray[]
    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[left + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];
 
    auto indexOfSubArrayOne = 0, indexOfSubArrayTwo = 0;
    int indexOfMergedArray = left;
 
    // Merge the temp arrays back into array[left..right]
    while (indexOfSubArrayOne < subArrayOne
           && indexOfSubArrayTwo < subArrayTwo) {
        if (leftArray[indexOfSubArrayOne]
            <= rightArray[indexOfSubArrayTwo]) {
            array[indexOfMergedArray]
                = leftArray[indexOfSubArrayOne];
            indexOfSubArrayOne++;
        }
        else {
            array[indexOfMergedArray]
                = rightArray[indexOfSubArrayTwo];
            indexOfSubArrayTwo++;
        }
        indexOfMergedArray++;
    }
 
    // Copy the remaining elements of
    // left[], if there are any
    while (indexOfSubArrayOne < subArrayOne) {
        array[indexOfMergedArray]
            = leftArray[indexOfSubArrayOne];
        indexOfSubArrayOne++;
        indexOfMergedArray++;
    }
 
    // Copy the remaining elements of
    // right[], if there are any
    while (indexOfSubArrayTwo < subArrayTwo) {
        array[indexOfMergedArray]
            = rightArray[indexOfSubArrayTwo];
        indexOfSubArrayTwo++;
        indexOfMergedArray++;
    }
    delete[] leftArray;
    delete[] rightArray;
}
 
void mergeSort(int array[], int const begin, int const end)
{
    if (begin >= end)
        return;
 
    int mid = begin + (end - begin) / 2;
    mergeSort(array, begin, mid);
    mergeSort(array, mid + 1, end);
    merge(array, begin, mid, end);
}
bool compareVal(int arr1[],int n1,int arr2[],int n2)
{
    mergeSort(arr2,0,n2-1);
    for(int i = 0;i<n1;i++)
    {
        if(arr1[i] != arr2[i])  return false;
    }
    return true;
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
        int *newSize;
        int *d_delElem;
        

        cudaHostAlloc(&curSize, sizeof(int), 0);
        cudaHostAlloc(&elemSize, sizeof(int), 0);
        cudaHostAlloc(&newSize, sizeof(int), 0);

        int h_a[maxSize];
        *curSize = getRandom(9,maxSize);
        int saveSize = *curSize;

        //Initialise Heap with some random values
        FillArray(h_a,*curSize);

       //heapify the heap
        buildHeap(h_a,*curSize);

        printf("%d. Initially the array is ",inivalid);
        printArray(h_a,*curSize);
        printf("\n");

       //check if satisfies the heap property
        if(checkHeap(h_a,*curSize)) inivalid++;

        cudaMalloc(&d_a,maxSize*sizeof(int)); 
        cudaMemcpy(d_a,h_a,maxSize * sizeof(int),cudaMemcpyHostToDevice);

        *elemSize = getRandom(*curSize,*curSize);
        int delElem[*elemSize];
        printf("No of elements to be deleted = %d, ",*elemSize);
        *newSize = *curSize - *elemSize;

        cudaMalloc(&d_delElem,(*elemSize)*sizeof(int));
        cudaMalloc(&lockArr,(*curSize)*sizeof(int));
        cudaMemset(lockArr,0,(*curSize)*sizeof(int));
    
        int block = ceil((float) *elemSize/1024);

        double starttime = rtclock();
        init<<<1,1>>>(); 
        Del_Elem<<<block,1024>>>(d_a,curSize,lockArr,elemSize,newSize,d_delElem);
        // Del_Elem<<<*elemSize,1>>>(d_a,curSize,lockArr,elemSize,newSize);
        cudaDeviceSynchronize();
        double endtime = rtclock();  
        printtime("Time: ", starttime, endtime);

        cudaMemcpy(delElem,d_delElem,*elemSize*sizeof(int),cudaMemcpyDeviceToHost);
        printf("Elements Deleted are ");
        printArray(delElem,*elemSize);
        printf("\n\n");

        bool res = compareVal(delElem,*elemSize,h_a,saveSize);
        cudaMemcpy(h_a,d_a,maxSize*sizeof(int),cudaMemcpyDeviceToHost);
        
        // printf("\nAfter Deletion the array is ");
        // printArray(h_a,*curSize);
        // printf("\n\n");
        
        if(checkHeap(h_a,*curSize) && res) {
            // printf("Valid\n");
            countvalid++;
        }
    }
    printf("\nInitial valid : %d",inivalid);
    printf("\nvalid : %d\n\n",countvalid);
    return 0;
}
