#include <iostream>

using namespace std;

__global__
void add10(float* arr) {
    // assume we have enough threads to have 1 / array element
    arr[threadIdx.x + blockIdx.x*blockDim.x] += 10.0f;
}

int main() {
    // allocate unified memory
    float *myArr;
    int ARRAY_SIZE = 12;
    cudaMallocManaged(&myArr, ARRAY_SIZE);

    // initialize array in unified memory
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        myArr[i] = i;
    }

    // launch kernel
    add10<<<1,ARRAY_SIZE>>>(myArr);
    cudaDeviceSynchronize();

    // print out the contents of the array
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        cout << myArr[i] << endl; 
    }
    cudaFree(myArr);
}