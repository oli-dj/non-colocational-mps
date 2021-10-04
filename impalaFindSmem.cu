#define CUDART_NAN_F            __int_as_float(0x7fffffff)
#include "math.h"

using namespace std;
 
__global__ void impalaFindSmem(const float* list, const float* dataEvent, const int listLength, const int dataLength, int* matches) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
    int temp = 1;

    //TODO: Implement shared memory
    //__shared__ float dataEventShared[1024];
    //dataEventShared = dataEvent;
       

    if(x < listLength) {
        // For each element in dataEvent
        for(int y=0; y<dataLength ; ++y){
        // Compare to entry in list and set temp to 0 if mismatch, except if NaN

            if(!isnan(dataEvent[y]) && (list[dataLength*x+y] != dataEvent[y])){
                temp = 0;
            }
        }
        
        matches[x] = temp;
    }
} 