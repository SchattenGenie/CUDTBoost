#include <iostream>
#include <stdio.h>
#include <ctime>

#define LOG_NUM_BANKS 5
#define NUM_BANKS 32
#define BLOCK_SIZE 64

void TotalPrescanGPU(const float * data, float * partial_sums, size_t data_size);
