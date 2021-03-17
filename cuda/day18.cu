#include <iostream>
#include "aoc_utils.hh"

const dim3 BLOCKS(1, 1, 1);
const dim3 THREADS(16, 16, 1);

const int STEPS = 100;

__device__ __managed__ int lights1[100][100];
__device__ __managed__ int lights2[100][100];

__device__ void next_state(int x, int y, int& state1, int& state2)
{
    int neighbors1 = 0;
    int neighbors2 = 0;
    for (int i = x - 1; i <= x + 1; ++i)
    {
        for(int j = y - 1; j <= y + 1; ++j)
        {
            if (i >= 0 && i < 100 && j >=0 && j < 100)
            {
                neighbors1 += lights1[i][j];
                neighbors2 += lights2[i][j];
            }
        }
    }
    state1 = ((neighbors1 == 3) || (neighbors1 == 4 && lights1[x][y] == 1));
    state2 = ((neighbors2 == 3) || (neighbors2 == 4 && lights2[x][y] == 1));
}

__device__ void one_step()
{
    int vals1[100];
    int vals2[100];
    int idx = 0;
    for (int row = threadIdx.x; row < 100; row += blockDim.x)
    {
        for(int col = threadIdx.y; col < 100; col += blockDim.y)
        {
            next_state(row, col, vals1[idx], vals2[idx]);
            ++idx;
        }
    }
    __syncthreads();
    idx = 0;
    for (int row = threadIdx.x; row < 100; row += blockDim.x)
    {
        for(int col = threadIdx.y; col < 100; col += blockDim.y)
        {
            lights1[row][col] = vals1[idx];
            lights2[row][col] = vals2[idx];
            if ((row == 0 && (col == 0 || col == 99)) ||
                (row == 99 && (col == 0 || col == 99)))
            {
                lights2[row][col] = 1;
            }
            ++idx;
        }
    }
    __syncthreads();
}

__device__ void count_lights(int* counts)
{
    __shared__ int totalCount1[256];
    __shared__ int totalCount2[256];
    int count1 = 0;
    int count2 = 0;
    for (int row = threadIdx.x; row < 100; row += blockDim.x)
    {
        for(int col = threadIdx.y; col < 100; col += blockDim.y)
        {
            count1 += lights1[row][col];
            count2 += lights2[row][col];
        }
    }
    int cacheIndex = threadIdx.x + blockDim.x * threadIdx.y;
    totalCount1[cacheIndex] = count1;
    totalCount2[cacheIndex] = count2;
    __syncthreads();

    reduction2D(totalCount1, cacheIndex);
    reduction2D(totalCount2, cacheIndex);

    if (cacheIndex == 0)
    {
        counts[0] = totalCount1[0];
        counts[1] = totalCount2[0];
    }
}

__global__ void update_lights(int* counts)
{
    for (int step = 0; step < STEPS; ++step)
    {
        one_step();
    }
    count_lights(counts);
}

int main()
{
    auto dataLines = data_lines("../data/day18.input.txt");

    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            lights1[i][j] = (dataLines[i][j] == '#') ? 1 : 0;
            lights2[i][j] = (dataLines[i][j] == '#') ? 1 : 0;
        }
    }
    lights2[0][0] = 1;
    lights2[0][99] = 1;
    lights2[99][0] = 1;
    lights2[99][99] = 1;

    int counts[2] = {0, 0};
    int* dev_counts;
    cudaMalloc(&dev_counts, 2 * sizeof(int));

    update_lights<<<BLOCKS,THREADS>>>(dev_counts);

    cudaMemcpy(&counts, dev_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Solution 1: " << counts[0] << std::endl;
    std::cout << "Solution 2: " << counts[1] << std::endl;

    cudaFree(dev_counts);
}
