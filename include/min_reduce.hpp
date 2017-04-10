#include <iostream>
#include <stdio.h>
#include <ctime>

#define BLOCK_SIZE 256
#define MAX_FLOAT 1.0e+127
#define WARP_SIZE 32

void TotalReduceGPU(const float * data, float * min, size_t data_size);
