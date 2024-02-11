    #include <cstdio>        // Added for printf() function 
    #include <sys/time.h>    // Added to get time of day
    #include <cuda.h>
    #include <fstream>
    #include <time.h>
    #include <iostream>
    #include <thrust/host_vector.h>
    #include <thrust/device_vector.h>
    #include <thrust/sort.h>

    using namespace std;
    //total size of the heap
    #define maxSize 1000000

    __global__ void Insert_Elem(int *heap,int *d_elements,int *curSize,int *elemSize){
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid < *elemSize)
        {
            heap[tid + *curSize] = d_elements[tid];
        }
    }

    void deleteElem(int *heap,int *curSize){
        //wrap raw pointer with a device_ptr
        thrust::device_ptr<int> d_vec(heap);
        //use device_ptr in thrust algorithms
        thrust::sort(d_vec, d_vec+*curSize);
        // cout << endl<<"Array after sorting"<<endl;
        // for(int i = 0;i<*curSize;i++){
        //     cout << d_vec[i] << " ";
        // }
        // cout << endl;
    }

    __global__ void delete_Elem(int *heap,int *d_elements,int *curSize,int *elemSize,int k){
       
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
            elements[i] = getRandom(1,maxSize*10);
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
        int *heap,*curSize,*elemSize;

        cudaHostAlloc(&curSize, sizeof(int), 0);
        cudaHostAlloc(&elemSize, sizeof(int), 0);

        int h_a[maxSize];
        *curSize = 0;

        cudaMalloc(&heap,maxSize*sizeof(int)); 
        cudaMemcpy(heap,h_a,maxSize * sizeof(int),cudaMemcpyHostToDevice);

        for(int lk = 0;lk<5;lk++)
        {
            do{
                *elemSize = getRandom(1,maxSize-*curSize);
            }while(*elemSize + *curSize > maxSize);
            
            int elements[*elemSize];
            
            FillArray(elements,*elemSize);

            printf("No of Inserted Elements are = %d\n",*elemSize);

            int *d_elements;
            cudaMalloc(&d_elements,*elemSize*sizeof(int));
            cudaMemcpy(d_elements,elements,*elemSize * sizeof(int),cudaMemcpyHostToDevice);

            double starttime = rtclock(); 

            int block = ceil((float) *elemSize/1024);
            Insert_Elem<<<block,1024>>>(heap,d_elements,curSize,elemSize);
            cudaDeviceSynchronize();
            double endtime = rtclock();
            *curSize = *curSize + *elemSize;  
            printtime("Insertion time: ", starttime, endtime); 


            starttime = rtclock();
            deleteElem(heap,curSize);
            endtime = rtclock();
            printtime("Sorting: ", starttime, endtime);
            cout << endl;
        }

        printf( " Over ");
        return 0;
    }
