__device__ inline size_t NoConflictIndex(size_t index);

__global__ void PrescanBlocks(float * out_data, const float * in_data, float * block_sums, const size_t data_size);

__global__ void AddBlockSums(float * data, const float * block_sums, const size_t data_size);

__host__ void PrescanBlockSums(float * block_sums, const size_t num_blocks);

void TotalPrescanGPU(const float * data, float * partial_sums, size_t data_size);

void TotalPrescanCPU(const float * data, float * partial_sums, size_t data_size);
