
__global__ void multiplyArray(const int* matches, const int* counts, const int N, const int M, int* counts_out){
    /*
    Embarrassingly simple parallel multiply where output and first matrix are 1D
     
    Inputs:
    matches: N length array of matches (ideally ones and zeros)
    counts: N*M length array of counts (integers)
    N: listLenght
    M: numFacies


    Input/Output:
    counts_out: Result of element-wise multiplication of 'a' with each "row" of 'b'.
    */
    
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // For length of list do
    if(x < N){
        for(int y = 0; y < M; y++){
            counts_out[N*y + x] = matches[x] * counts[N*y + x];
        }
    }
} 