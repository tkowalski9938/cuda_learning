#include <iostream>
#include <string>

class Managed {
    public:
        void* operator new(size_t len) {
            void* ptr;
            cudaMallocManaged(&ptr, len);
            //cudaDeviceSynchronize();
            return ptr;
        }
        void operator delete(void* ptr) {
            //cudaDeviceSynchronize();
            cudaFree(ptr);
        }
};

class Square : public Managed {
    int size;
    public:
        float **arr;
        Square(int size) : size{size} {
            cudaMallocManaged(&arr, size*sizeof(float*));
            for (int i = 0; i < size; ++i) {
                cudaMallocManaged(&(arr[i]), size*sizeof(float));
                  //std::cout << "right here" << std::endl;
            }
        }
        ~Square() {
            for (int i = 0; i < size; ++i) {
                cudaFree(arr[i]);
            }
            cudaFree(arr);
        }

        int getSize() {return size;}

        void initialize() {
            for (int i = 0; i < size; ++i) {
                for (int k = 0; k < size; ++k) {
                    arr[i][k] = 1.0f;
                }
            }
        }

        void print() {
            for (int i = 0; i < size; ++i) {
                std::cout << "Row " << i << ": ";
                for (int k = 0; k < size; ++k) {
                    std::cout << arr[i][k] << ", ";
                }
                std::cout << std::endl;
            }
        }
};

__global__
void modifySquare(Square* sqr, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    // each element is multiplied by (column+1), divided by (row+1)

    // grid-stride loop
    for (; row < size; row += (gridDim.y * blockDim.y)) {
        for (; column < size; column += (gridDim.x * blockDim.x)) {
            //sqr->arr[row][column] = (column+1) / (row+1);
            sqr->arr[row][column] = row;
        }
    }
}

int main() {
    int N = 5;
    Square* mySqr = new Square{N};

    // initialize the fields
    mySqr->initialize();

    // launch kernel
    uint threadDim = 16;
    uint blocksDim = (N + threadDim - 1) / N;
    dim3 blocks{blocksDim, blocksDim, 1};
    dim3 threads{threadDim, threadDim, 1};
    
    modifySquare<<<blocks, threads>>>(mySqr, mySqr->getSize());

    cudaError_t err = cudaGetLastError();
    if (err) {
         std::string myError{cudaGetErrorName(err)};
         std::cout << "The following error has been detected: " << myError << std::endl;
    }
    cudaDeviceSynchronize();
    
    // print out the result
    mySqr->print();

    delete mySqr;
}