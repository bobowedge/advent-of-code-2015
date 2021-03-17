// nvcc -g -o day01.exe day01.cu
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "aoc_utils.hh" // For reduction

// Somewhat arbitrary values for THREADS and BLOCKS
const size_t THREADS = 32;
const size_t BLOCKS = 32;

/**
 * \brief Count the floors from the instructions
 * 
 * \param instructions String of '(' and ')'
 * \param result Output sum for each block
 * \param N Length of instructions
 */
__global__ void count_floors(char *instructions, int64_t *result, 
    const size_t N)
{
    // Core device loop
    int64_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    // Sum for this thread
    int64_t sum = 0;
    while (threadIndex < N)
    {
        if (instructions[threadIndex] == '(')
        {
            ++sum;
        }
        else
        {
            --sum;
        }
        threadIndex += blockDim.x * gridDim.x;
    }

    // Block shared memory array
    __shared__ int64_t cache[THREADS];
    const int64_t cacheIndex = threadIdx.x;
    cache[cacheIndex] = sum;

    // Sync every thread in this block
    __syncthreads();

    // Reduce cache to a single value
    reduction(cache, cacheIndex);

    if (cacheIndex == 0)
    {
        result[blockIdx.x] = cache[0];
    }
}

int main()
{
    // Read the data into a C++ string
    std::ifstream data("../data/day01.input.txt");
    std::string instructions;
    std::getline(data, instructions);

    // Get the size of the string
    const size_t N = instructions.size();

    // Stores the result for each block
    std::vector<int64_t> partial_results(BLOCKS);

    // Device string for instructions
    char *dev_instructions;
    // Device instructions size
    size_t dev_N_size = N * sizeof(char);
    // Allocate the memory on the GPU
    cudaMalloc((void **)&dev_instructions, dev_N_size);

    // Device partial results
    int64_t *dev_partial_results;
    // Device partial results size   
    size_t dev_pr_size = partial_results.size() * sizeof(int64_t);
    // Allocate the memory on the GPU    
    cudaMalloc((void **)&dev_partial_results, dev_pr_size);

    // Copy the instruction onto the GPU
    cudaMemcpy(dev_instructions, instructions.c_str(), dev_N_size, cudaMemcpyHostToDevice);

    // Run the device code
    count_floors<<<BLOCKS,THREADS>>>(dev_instructions, dev_partial_results, N);

    // Copy the results back from the GPU
    cudaMemcpy(partial_results.data(), dev_partial_results, dev_pr_size, cudaMemcpyDeviceToHost);

    // Sum the floors for each block
    int64_t floors = std::accumulate(partial_results.begin(), 
                                     partial_results.end(), 
                                     static_cast<int64_t>(0));

    // Print the solution
    std::cout << "Solution 1: " << floors << std::endl;

    // Free the device memory
    cudaFree (dev_instructions);
    cudaFree (dev_partial_results);
}
