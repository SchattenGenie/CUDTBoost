#include <iostream>
#include <stdio.h>
#include <ctime>

#define BLOCK_SIZE 256
#define MAX_FLOAT 1.0e+127
#define WARP_SIZE 32

inline __device__ float Min(const float first, const float second) {
        if (first < second) {
                return first;
        }
        return second;
}

__global__ void ReduceBlocks(float * out_data, const float * in_data, const size_t data_size) {
        extern __shared__ float in_data_shared[];

        size_t thread_id_local = threadIdx.x;
        size_t thread_id_global = blockIdx.x * 2 * blockDim.x + thread_id_local;

        float left_data = MAX_FLOAT;
        float right_data = MAX_FLOAT;
        if (thread_id_global < data_size) {
                left_data = in_data[thread_id_global];
        }

        if (thread_id_global + blockDim.x < data_size) {
                right_data = in_data[thread_id_global + blockDim.x];
        }

        in_data_shared[thread_id_local] = Min(left_data, right_data);

        __syncthreads();

        for (size_t stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
                if (thread_id_local < stride) {
                        in_data_shared[thread_id_local] = Min(in_data_shared[thread_id_local], in_data_shared[thread_id_local + stride]);
                }
                __syncthreads();
        }

        if (thread_id_local < WARP_SIZE) {
                in_data_shared[thread_id_local] = Min(in_data_shared[thread_id_local], in_data_shared[thread_id_local + 32]);
                in_data_shared[thread_id_local] = Min(in_data_shared[thread_id_local], in_data_shared[thread_id_local + 16]);
                in_data_shared[thread_id_local] = Min(in_data_shared[thread_id_local], in_data_shared[thread_id_local + 8]);
                in_data_shared[thread_id_local] = Min(in_data_shared[thread_id_local], in_data_shared[thread_id_local + 4]);
                in_data_shared[thread_id_local] = Min(in_data_shared[thread_id_local], in_data_shared[thread_id_local + 2]);
                in_data_shared[thread_id_local] = Min(in_data_shared[thread_id_local], in_data_shared[thread_id_local + 1]);
        }

        if (thread_id_local == 0) {
                out_data[blockIdx.x] = in_data_shared[0];
        }
}

__host__ float ReduceBlockResults(const float * block_reduces, const size_t num_blocks) {
        float min = MAX_FLOAT;
        for (size_t idx = 0; idx < num_blocks; ++idx) {
                min = std::min(block_reduces[idx], min);
        }

        return min;
}

void TotalReduceGPU(const float * data, float * min, size_t data_size) {
        float * d_data;
        float * d_block_mins;

        size_t num_blocks = ((data_size + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE));
        size_t shared_size = BLOCK_SIZE * sizeof(float);

        float * block_mins = (float *) malloc(num_blocks * sizeof(float));

        cudaMalloc(&d_data, data_size * sizeof(float));
        cudaMalloc(&d_block_mins, num_blocks * sizeof(float));

        cudaMemcpy(d_data, data, data_size * sizeof(float), cudaMemcpyHostToDevice);
        ReduceBlocks<<<num_blocks, BLOCK_SIZE, shared_size>>>(d_block_mins, d_data, data_size);

        cudaMemcpy(block_mins, d_block_mins, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
        *min = ReduceBlockResults(block_mins, num_blocks);
        cudaFree(d_block_mins);
        cudaFree(d_data);
        free(block_mins);
}

void TotalReduceCPU(const float * data, float * min, size_t data_size) {
        float min_here = MAX_FLOAT;
        for (size_t idx = 0; idx < data_size; ++idx) {
                min_here = std::min(min_here, data[idx]);
        }

        *min = min_here;
}

int main(int argc, char * argv[]) {
        float * data;
        float min;
        size_t logsize = atoi(argv[1]);
        size_t num_elements = (1 << logsize);
        // size_t num_elements = 7;
        data = (float *) malloc(num_elements * sizeof(float));

        for (size_t idx = 0; idx < num_elements; ++idx) {
                data[idx] = -1.0 * idx * idx + 100.0;
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
                TotalReduceGPU(data, &min, num_elements);
                float milliseconds = float(clock () - begin_time) / 1000;
                // cudaEventRecord(stop);

                // cudaEventSynchronize(stop);
                // float milliseconds = 0;
                // cudaEventElapsedTime(&milliseconds, start, stop);
                // std::cout << "GPU run took " << milliseconds << " ms" << std::endl;
                runtimes[run] = milliseconds;
                gpu_mean += milliseconds / num_runs;
        }
        std::cout << runtimes[0] << std::endl;
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
                TotalReduceCPU(data, &min, num_elements);
                float milliseconds = float(clock () - begin_time) / 1000;
                // cudaEventRecord(stop);

                // cudaEventSynchronize(stop);
                // float milliseconds = 0;
                // cudaEventElapsedTime(&milliseconds, start, stop);
                // std::cout << "GPU run took " << milliseconds << " ms" << std::endl;
                runtimes[run] = milliseconds;
                cpu_mean += milliseconds / num_runs;
        }
        std::cout << runtimes[0] << std::endl;
        for (size_t run = 0; run < num_runs; ++run) {
                cpu_std += (cpu_mean - runtimes[run]) * (cpu_mean - runtimes[run]) / num_runs;
        }
        cpu_std = sqrt(cpu_std);

        std::cout << num_elements << " " << gpu_mean << " " << gpu_std << " " << cpu_mean << " " << cpu_std << std::endl;

        free(data);

        return 0;
}
