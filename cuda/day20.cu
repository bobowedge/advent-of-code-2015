#include <cstdint>
#include <iostream>
#include <chrono>
#include <vector>

const uint64_t PUZZLE_INPUT = 36000000;
const uint64_t MAX_HOUSES1 = PUZZLE_INPUT/10;
const uint64_t MAX_HOUSES2 = PUZZLE_INPUT/11 + 1;

const uint64_t BLOCKS = 1024;
const uint64_t THREADS = 256;

__device__ __managed__ uint64_t* houses1;
__device__ __managed__ uint64_t* houses2;

__global__ void reset_houses()
{
    for(uint64_t house = threadIdx.x + blockIdx.x * blockDim.x;
        house < MAX_HOUSES1;
        house += gridDim.x * blockDim.x)
    {
        houses1[house] = 0;
        if (house < MAX_HOUSES2)
        {
            houses2[house] = 0;
        }
    }
}

__global__ void deliver_by_house()
{
    for(uint64_t house = threadIdx.x + blockIdx.x * blockDim.x;
        house < MAX_HOUSES1;
        house += gridDim.x * blockDim.x)
    {
        uint64_t presents1 = 0;
        uint64_t presents2 = 0;
        for (uint64_t j = 1; j <= house; ++j)
        {
            if ((house % j) == 0)
            {
                presents1 += 10 * j;
                if (j >= house/50 && j < MAX_HOUSES2)
                {
                    presents2 += 11 * j;
                }
            }
        }
        houses1[house] = presents1;
        if (house < MAX_HOUSES2)
        {
            houses2[house] = presents2;
        }
    }
}

__global__ void deliver_by_elf()
{
    for(uint64_t elf = threadIdx.x + blockIdx.x * blockDim.x;
        elf < MAX_HOUSES1;
        elf += gridDim.x * blockDim.x)
    {
        if (elf == 0)
            continue;
        uint64_t presents1 = 10 * elf;
        for (uint64_t j = elf; j < MAX_HOUSES1; j += elf)
        {
            if (houses1[j] + presents1 >= PUZZLE_INPUT)
                atomicExch(&houses1[j], PUZZLE_INPUT);
            else
                atomicAdd(&houses1[j], presents1);
        }
        uint64_t presents2 = 11 * elf;
        for (uint64_t j = elf; j <= 50 * elf && j < MAX_HOUSES2; j += elf)
        {
            if (houses2[j] + presents2 >= PUZZLE_INPUT)
                atomicExch(&houses2[j], PUZZLE_INPUT);
            else
                atomicAdd(&houses2[j], presents2);
        }
    }
}

uint64_t check_solution1()
{
    for (uint64_t i = 1; i < MAX_HOUSES1; ++i)
    {
        if (houses1[i] >= PUZZLE_INPUT)
        {
            return i;
        }
    }
    return ULONG_MAX;
}

uint64_t check_solution2()
{
    for (uint64_t i = 1; i < MAX_HOUSES2; ++i)
    {
        if (houses2[i] >= PUZZLE_INPUT)
        {
            return i;
        }
    }
    return ULONG_MAX;
}

int main()
{
    cudaMallocManaged(&houses1, MAX_HOUSES1 * sizeof(uint64_t));
    cudaMallocManaged(&houses2, MAX_HOUSES2 * sizeof(uint64_t));

    reset_houses<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();

    // std::vector<uint64_t> blocks = {32, 64, 128, 256, 512, 1024};
    // std::vector<uint64_t> threads = {128, 256};

    auto start = std::chrono::high_resolution_clock::now();
    deliver_by_house<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << BLOCKS << " blocks, " << THREADS << " threads, ";
    std::cout << std::chrono::duration<double>(end - start).count() << " seconds\n";

    uint64_t soln1 = check_solution1();
    uint64_t soln2 = check_solution2();
    std::cout << "Solution 1: " << soln1 << std::endl;
    std::cout << "Solution 2: " << soln2 << std::endl;

    reset_houses<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();

    start = std::chrono::high_resolution_clock::now();
    deliver_by_elf<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    std::cout << BLOCKS << " blocks, " << THREADS << " threads, ";
    std::cout << std::chrono::duration<double>(end - start).count() << " seconds\n";

    soln1 = check_solution1();
    soln2 = check_solution2();
    std::cout << "Solution 1: " << soln1 << std::endl;
    std::cout << "Solution 2: " << soln2 << std::endl;

    cudaFree(houses1);
    cudaFree(houses2);
}

