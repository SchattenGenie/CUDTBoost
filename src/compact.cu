#include <iostream>
#include <stdio.h>
#include <cuda.h>
#define BLOCK_SIZE 256

__global__ void d_compact(float * d_classDistrTable, size_t * d_addr, bool * d_flag, float * d_buffer, size_t data_size) {
    
    unsigned int iGlobal = blockIdx.x * (blockDim.x << 2) + threadIdx.x;

    if (iGlobal < data_size && d_flag[iGlobal] == true) {
        d_buffer[d_addr[iGlobal]] = d_classDistrTable[iGlobal];
    }
    iGlobal += blockDim.x;

    if (iGlobal < data_size && d_flag[iGlobal] == true) {
        d_buffer[d_addr[iGlobal]] = d_classDistrTable[iGlobal];
    }
    iGlobal += blockDim.x;

    if (iGlobal < data_size && d_flag[iGlobal] == true) {
        d_buffer[d_addr[iGlobal]] = d_classDistrTable[iGlobal];
    }
    iGlobal += blockDim.x;

    if (iGlobal < data_size && d_flag[iGlobal] == true) {
        d_buffer[d_addr[iGlobal]] = d_classDistrTable[iGlobal];
    }


}

void compactCPU(float * h_classDistrTable, size_t * h_addr, bool * h_flag, size_t data_size) {
    float * h_buffer;
    h_buffer = (float *) malloc(data_size * sizeof(float));
    for (size_t idx = 0; idx < data_size; ++idx) {
        if (h_flag[idx] == true) {
            h_buffer[h_addr[idx]] = h_classDistrTable[idx];
        };   
    }
    memcpy(h_classDistrTable, h_buffer, sizeof(float) * data_size);
    free(h_buffer);
}


void compactGPU(float * h_classDistrTable, size_t * h_addr, bool * h_flag, size_t data_size) {
    
    size_t num_blocks = ((data_size + 2 * BLOCK_SIZE - 1) / (BLOCK_SIZE));
    float * d_classDistrTable;
    size_t * d_addr;
    bool * d_flag;
    float * d_buffer;


    cudaMalloc(&d_classDistrTable, sizeof(float) * data_size);
    cudaMalloc(&d_addr, sizeof(size_t) * data_size);
    cudaMalloc(&d_flag, sizeof(bool) * data_size);
    cudaMalloc(&d_buffer, sizeof(float) * data_size); //change it

    cudaMemcpy(d_classDistrTable, h_classDistrTable, data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_addr, h_addr, data_size * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, h_flag, data_size * sizeof(bool), cudaMemcpyHostToDevice);

    d_compact<<<num_blocks, BLOCK_SIZE>>>(d_classDistrTable,  d_addr, d_flag, d_buffer, data_size);

    cudaMemcpy(h_classDistrTable, d_buffer, data_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_classDistrTable);
    cudaFree(d_addr);
    cudaFree(d_flag);
    cudaFree(d_buffer);
}

bool randomBool() {
  return rand() % 2 == 1;
}

int main(int argc, char * argv[]) {
    float * classDistrTable;
    size_t * addr;
    bool * flag;
    size_t data_size = atoi(argv[1]);

    classDistrTable = (float *) malloc(data_size * sizeof(float));
    addr = (size_t *) malloc(data_size * sizeof(size_t));
    flag = (bool *) malloc(data_size * sizeof(bool));    

    for (size_t idx = 0; idx < data_size; ++idx) {
        classDistrTable[idx] = 1.0 * idx;
        addr[idx] = (idx + rand()) % data_size;
        flag[idx] = randomBool();
    }
    float milliseconds = 0;
    float milliseconds_squared = 0;
    float diff = 0;
    int N = 100;
    for (size_t idx = 0; idx < N; ++idx){
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        const clock_t begin_time = clock();
        compactGPU(classDistrTable, addr, flag, data_size);
        diff = float(clock() - begin_time) / 1000;
        milliseconds += diff;
        milliseconds_squared += diff * diff;
    }
    std::cout << "mean=" << milliseconds / N << " std=" << (milliseconds_squared - milliseconds * milliseconds / N) / N  << std::endl;


    milliseconds = 0;
    milliseconds_squared = 0;
    N = 100;
    for (size_t idx = 0; idx < N; ++idx){
        const clock_t begin_time = clock();
        compactCPU(classDistrTable, addr, flag, data_size);
        diff = float(clock() - begin_time) / 1000;
        milliseconds += diff;
        milliseconds_squared += diff * diff;
    }
    std::cout << "mean=" << milliseconds / N << " std=" << (milliseconds_squared - milliseconds * milliseconds / N) / N  << std::endl;


	return 0;
}
