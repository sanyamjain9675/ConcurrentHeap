#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <iostream>
#include <thrust/sort.h>
#include <sys/time.h>
#include <time.h>
#include <algorithm>
#include <vector>


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

int main(void)
{
    thrust::host_vector<int> h_vec(1000000);
    // vector <int> h_vec(1000000)
    std::generate(h_vec.begin(), h_vec.end(), rand);

    thrust::device_vector<int> d_vec = h_vec;
    
    double starttime = rtclock(); 
    std::sort(h_vec.begin(), h_vec.end());
    double endtime = rtclock();  
    printtime("Serial Sorting time: ", starttime, endtime);

    starttime = rtclock(); 
    thrust::sort(d_vec.begin(), d_vec.end());
    endtime = rtclock();  
    printtime("Parallel Sorting time: ", starttime, endtime);

    return 0;
}
