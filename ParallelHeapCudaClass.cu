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

    __global__ void delete_Elem(int *heap,int *d_elements,int *curSize,int *elemSize,int k){
       
    }

    int getRandom(int lower, int upper)
    {
        int num = (rand() % (upper - lower + 1)) + lower;
        return num;  
    }

    void printArray(thrust::device_vector<int> arr,int size)
    {
        for(int i = 0;i<size;i++)
            printf("%d, ",arr[i]);
    }

    void FillArray(thrust::host_vector<int> &elements,int size)
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

    //Insert If only key is there
    __global__ void Insert_Elem(thrust::device_vector<int> &heap,int *curSize,thrust::device_vector<int> &d_elements,int *elemSize)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid < *elemSize)
        {
            heap[tid + *curSize] = d_elements[tid];
        }
    }

    //Insert if both key and values are there
    __global__ void Insert_Elem(thrust::device_vector<int> &d_val, thrust::device_vector<int> &heap_val,thrust::device_vector<int> &heap,int *curSize,thrust::device_vector<int> &d_elements,int *elemSize)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if(tid < *elemSize)
        {
            heap[tid + *curSize] = d_elements[tid];
            heap_val[tid + *curSize] = d_val[tid];
        }
    }

    class Heap{
        private:
        thrust::device_vector<int> heap,heap_val;
        int *curSize;
        bool isSorted,isType;
        public:
        Heap(){
            srand(time(0));
            // cout << "Constructor Called" << endl;
            cudaHostAlloc(&curSize, sizeof(int), 0);
            // cudaMalloc(&heap,maxSize*sizeof(int));
            // cudaMalloc(&heap_val,maxSize*sizeof(int));
            heap.resize(maxSize);
            heap_val.resize(maxSize);
            isSorted = false;
            isType = 1;
            // *curSize = 0;
        }

        int getSize(){
            return *curSize;
        }

        //Insert If only key is there
        void insert(thrust::host_vector<int> &elements,int size){
            isType = 1;
            isSorted = false;
            int *elemSize;
            thrust::device_vector<int> d_elements = elements;

            cudaMalloc(&elemSize,sizeof(int));
            cudaMemcpy(elemSize,&size,sizeof(int),cudaMemcpyHostToDevice);
            
            int block = ceil((float) size/1024);
            Insert_Elem<<<block,1024>>>(heap,curSize,d_elements,elemSize);
            cudaDeviceSynchronize();
            *curSize = *curSize + size; 
        }

        //Insert if both key and values are there
        void insert(thrust::host_vector<int> &elements,thrust::host_vector<int> &val,int size){
            isType = 2;
            isSorted = false;
            int *elemSize;
            thrust::device_vector<int> d_elements = elements;
            thrust::device_vector<int> d_val = val;

            cudaMalloc(&elemSize,sizeof(int));
            cudaMemcpy(elemSize,&size,sizeof(int),cudaMemcpyHostToDevice);
            
            int block = ceil((float) size/1024);
            Insert_Elem<<<block,1024>>>(d_val,heap_val,heap,curSize,d_elements,elemSize);
            cudaDeviceSynchronize();
            *curSize = *curSize + size; 
        }


        void deleteElem(){
            
            //wrap raw pointer with a device_ptr
            // thrust::device_ptr<int> d_vec(heap);
            // thrust::device_ptr<int> d_values(heap_val);
            //use device_ptr in thrust algorithms
            thrust::sort_by_key(heap,heap+*curSize,heap_val);
            isSorted = true;
            printArray(heap,*curSize);
            // cout << endl<<"Array after sorting"<<endl;
            // for(int i = 0;i<*curSize;i++){
            //     cout << d_vec[i] << "->"<< d_values[i] << " ; ";
            // }
            // cout << endl;
        }

    };
    

    int main() {
        
        Heap hp;

        for(int lk = 0;lk<5;lk++)
        {
            int elemSize;
            do{
                elemSize = getRandom(1,maxSize-hp.getSize());
            }while(elemSize + hp.getSize() > maxSize);
            
            // int elements[elemSize];
            thrust::host_vector<int> elements(elemSize);
            FillArray(elements,elemSize);

            printf("No of Inserted Elements are = %d\n",elemSize);
            double starttime = rtclock(); 
            hp.insert(elements,elemSize);
            double endtime = rtclock(); 
            printtime("Insertion time: ", starttime, endtime); 


            starttime = rtclock();
            hp.deleteElem();
            endtime = rtclock();
            printtime("Sorting: ", starttime, endtime);
            cout << endl;
        }

        printf( " Over ");
        return 0;
    }
