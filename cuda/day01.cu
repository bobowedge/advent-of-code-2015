#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "aoc_utils.hh"

const size_t THREADS = 32;
const size_t BLOCKS = 32;

__global__ void count_floors(char *instructions, int64_t *floors, 
    const size_t N)
{
    __shared__ int64_t cache[THREADS];
    int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t cacheIndex = threadIdx.x;
    int64_t temp = 0;
    while (tid < N)
    {
        if (instructions[tid] == '(')
        {
            temp += 1;
        }
        else
        {
            temp -= 1;
        }
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    reduction(cache, cacheIndex);

    if (cacheIndex == 0)
    {
        floors[blockIdx.x] = cache[0];
    }
}

int main()
{
    std::ifstream data("../data/day01.input.txt");
    std::string instructions;
    std::getline(data, instructions);

    const size_t N = instructions.size();

    std::vector<int64_t> partial_floors(BLOCKS);

    char *dev_instructions;
    int64_t *dev_partial_floors;

    size_t dev_i_size = N * sizeof(char);
    size_t dev_pf_size = partial_floors.size() * sizeof(int64_t);
    
    cudaMalloc((void **)&dev_instructions, dev_i_size);
    cudaMalloc((void **)&dev_partial_floors, dev_pf_size);
    cudaMemcpy(dev_instructions, instructions.c_str(), dev_i_size, cudaMemcpyHostToDevice);

    count_floors<<<BLOCKS,THREADS>>>(dev_instructions, dev_partial_floors, N);

    cudaMemcpy(partial_floors.data(), dev_partial_floors, dev_pf_size, cudaMemcpyDeviceToHost);

    int64_t floors = std::accumulate(partial_floors.begin(), 
                                     partial_floors.end(), 
                                     static_cast<int64_t>(0));

    std::cout << "Solution 1: " << floors << std::endl;

    cudaFree (dev_instructions);
    cudaFree (dev_partial_floors);
}
