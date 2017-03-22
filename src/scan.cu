#include <iostream>
#include <stdio.h>
#include <ctime>

#define LOG_NUM_BANKS 5
#define NUM_BANKS 32
#define BLOCK_SIZE 64

__device__ inline size_t NoConflictIndex(size_t index) {
    return index;
    // return index + (index >> LOG_NUM_BANKS);
}

__global__ void PrescanBlocks(float * out_data, const float * in_data, float * block_sums, const size_t data_size) {
    // keeps all the in_data during processing
    extern __shared__ float in_data_shared[];

    size_t thread_id_local = threadIdx.x;
    size_t offset = 1;
    size_t thread_id_global = blockIdx.x * blockDim.x + thread_id_local;
    if (thread_id_global >= data_size) {
        return;
    }

    in_data_shared[NoConflictIndex(2 * thread_id_local)] = in_data[2 * thread_id_global];
    in_data_shared[NoConflictIndex(2 * thread_id_local + 1)] = in_data[2 * thread_id_global + 1];

    for (size_t level_size = 2 * blockDim.x >> 1; level_size > 0; level_size >>= 1) {
        __syncthreads();
        if (thread_id_local < level_size) {
            size_t left_son_idx = offset * (2 * thread_id_local + 1) - 1;
            size_t parent_idx = offset * (2 * thread_id_local + 2) - 1;
            in_data_shared[NoConflictIndex(parent_idx)] += in_data_shared[NoConflictIndex(left_son_idx)];
        }

        offset *= 2;
    }

    if (thread_id_local == 0) {
        block_sums[blockIdx.x] = in_data_shared[NoConflictIndex(blockDim.x * 2 - 1)];
        in_data_shared[NoConflictIndex(blockDim.x * 2 - 1)] = 0;
    }

    for (size_t level_size = 1; level_size < 2 * blockDim.x; level_size *= 2) {
        offset >>= 1;
        __syncthreads();

        if (thread_id_local < level_size) {
            size_t left_son_idx = offset * (2 * thread_id_local + 1) - 1;
            size_t parent_idx = offset * (2 * thread_id_local + 2) - 1;

            float left_son_value = in_data_shared[NoConflictIndex(left_son_idx)];
            in_data_shared[NoConflictIndex(left_son_idx)] = in_data_shared[NoConflictIndex(parent_idx)];
            in_data_shared[NoConflictIndex(parent_idx)] += left_son_value;
        }
    }

    __syncthreads();

    out_data[2 * thread_id_global] = in_data_shared[NoConflictIndex(2 * thread_id_local)];
    out_data[2 * thread_id_global + 1] = in_data_shared[NoConflictIndex(2 * thread_id_local + 1)];
}

__global__ void AddBlockSums(float * data, const float * block_sums, const size_t data_size) {
    __shared__ float this_block_sum;

    size_t thread_id_local = threadIdx.x;
    size_t thread_id_global = blockIdx.x * blockDim.x + thread_id_local;

    if (thread_id_global >= data_size) {
        return;
    }

    if (thread_id_local == 0) {
        this_block_sum = block_sums[blockIdx.x];
    }
    __syncthreads();

    data[thread_id_global] += this_block_sum;
}

__host__ void PrescanBlockSums(float * block_sums, const size_t num_blocks) {
    float sum = block_sums[0];
    block_sums[0] = 0;
    float keep;
    for (size_t block_id = 1; block_id < num_blocks; ++block_id) {
        keep = block_sums[block_id];
        block_sums[block_id] = sum;
        sum += keep;
    }
}

void TotalPrescanGPU(const float * data, float * partial_sums, size_t data_size) {
    float * d_data;
    float * d_partial_sums;
    float * d_block_sums;
    float * block_sums;

    size_t num_blocks = ((data_size + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE));
    size_t shared_size = ((2 * BLOCK_SIZE + NUM_BANKS - 1) / NUM_BANKS + BLOCK_SIZE) * 2 * sizeof(float);

    block_sums = (float *) malloc(num_blocks * sizeof(float));

    cudaMalloc(&d_data, data_size * sizeof(float));
    cudaMalloc(&d_partial_sums, data_size * sizeof(float));
    cudaMalloc(&d_block_sums, num_blocks * sizeof(float));

    cudaMemcpy(d_data, data, data_size * sizeof(float), cudaMemcpyHostToDevice);
    PrescanBlocks<<<num_blocks, BLOCK_SIZE, shared_size>>>(d_partial_sums, d_data, d_block_sums, data_size);

    cudaMemcpy(block_sums, d_block_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
    PrescanBlockSums(block_sums, num_blocks);
    cudaMemcpy(d_block_sums, block_sums, num_blocks * sizeof(float), cudaMemcpyHostToDevice);

    AddBlockSums<<<num_blocks, 2 * BLOCK_SIZE>>>(d_partial_sums, d_block_sums, data_size);

    cudaMemcpy(partial_sums, d_partial_sums, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_block_sums);
    cudaFree(d_partial_sums);
    cudaFree(d_data);
    free(block_sums);
}

void TotalPrescanCPU(const float * data, float * partial_sums, size_t data_size) {
    float sum = 0.0;
    for (size_t idx = 0; idx < data_size; ++idx) {
        partial_sums[idx] = sum;
        sum += data[idx];
    }
}


int main(int argc, char * argv[]) {
    float * data;
    float * partial_sums;
    size_t logsize = atoi(argv[1]);
    size_t num_elements = (1 << logsize);

    data = (float *) malloc(num_elements * sizeof(float));
    partial_sums = (float *) malloc(num_elements * sizeof(float));

    for (size_t idx = 0; idx < num_elements; ++idx) {
        data[idx] = 1.0 * idx;
    }

    size_t num_runs = 100;
    float runtimes[100];
    float gpu_mean = 0.0;
    float gpu_std = 0.0;
    for (size_t run = 0; run < num_runs; ++run) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        // cudaEventRecord(start);
        const clock_t begin_time = clock();
        TotalPrescanGPU(data, partial_sums, num_elements);
        float milliseconds = float(clock () - begin_time) / 1000;
        // cudaEventRecord(stop);

        // cudaEventSynchronize(stop);
        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // std::cout << "GPU run took " << milliseconds << " ms" << std::endl;
        runtimes[run] = milliseconds;
        gpu_mean += milliseconds / num_runs;
    }

    for (size_t run = 0; run < num_runs; ++run) {
        gpu_std += (gpu_mean - runtimes[run]) * (gpu_mean - runtimes[run]) / num_runs;
    }
    gpu_std = sqrt(gpu_std);

    /*
    float true_answer = 0.0;
    bool correct = true;
    for (size_t idx = 0; idx < num_elements - 1; ++idx) {
        true_answer += idx;
        if (true_answer != partial_sums[idx + 1]) {
            correct = false;
            std::cout << idx << " " << partial_sums[idx + 1] << " " << true_answer << std::endl;
        }
    }

    if (!correct) {
        std::cout << "incorrect" << std::endl;
    }
    */

    float cpu_mean = 0.0;
    float cpu_std = 0.0;
    for (size_t run = 0; run < num_runs; ++run) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        // cudaEventRecord(start);
        const clock_t begin_time = clock();
        TotalPrescanCPU(data, partial_sums, num_elements);
        float milliseconds = float(clock () - begin_time) / 1000;
        // cudaEventRecord(stop);

        // cudaEventSynchronize(stop);
        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // std::cout << "GPU run took " << milliseconds << " ms" << std::endl;
        runtimes[run] = milliseconds;
        cpu_mean += milliseconds / num_runs;
    }

    for (size_t run = 0; run < num_runs; ++run) {
        cpu_std += (cpu_mean - runtimes[run]) * (cpu_mean - runtimes[run]) / num_runs;
    }
    cpu_std = sqrt(cpu_std);

    std::cout << num_elements << " " << gpu_mean << " " << gpu_std << " " << cpu_mean << " " << cpu_std << std::endl;

    free(data);
    free(partial_sums);

    return 0;
}
