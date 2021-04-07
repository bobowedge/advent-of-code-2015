#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

// Given number of input presents
const uint32_t PUZZLE_INPUT = 36000000;
// Maximum house number for Part 1
const uint32_t MAX_HOUSES1 = PUZZLE_INPUT/10;
// Maximum house number for Part 2
const uint32_t MAX_HOUSES2 = PUZZLE_INPUT/11 + 1;

// House stack for Part 1
__device__ __managed__ uint32_t* houses1;
// House stack for Part 2
__device__ __managed__ uint32_t* houses2;

// Reset the house stacks to 0
__global__ void reset_houses()
{
    for(uint32_t house = threadIdx.x + blockIdx.x * blockDim.x;
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

// Set the present for house (Part 1)
__device__ void set_house1(uint32_t house, uint32_t presents)
{
    if (houses1[house] + presents >= PUZZLE_INPUT)
    {
        atomicExch(&houses1[house], PUZZLE_INPUT);
    }
    else
    {
        atomicAdd(&houses1[house], presents);
    }
}

// Set the present for house (Part 2)
__device__ void set_house2(uint32_t house, uint32_t presents)
{
    if (houses2[house] + presents >= PUZZLE_INPUT)
    {
        atomicExch(&houses2[house], PUZZLE_INPUT);
    }
    else
    {
        atomicAdd(&houses2[house], presents);
    }
}

// Deliver the present house by house
__global__ void deliver_by_house()
{
    // Loop over the houses (starting at 1)
    for(uint32_t house = 1 + threadIdx.x + blockIdx.x * blockDim.x;
        house < MAX_HOUSES1;
        house += gridDim.x * blockDim.x)
    {
        // Loop over the possible elfs for this house
        for (uint32_t elf = 1; elf <= house; ++elf)
        {
            // Does this elf deliver here (i.e. is elf a factor of house)?
            if ((house % elf) == 0)
            {
                // Set presents for houses1
                set_house1(house, 10 * elf);
                // Does this elf deliver here (Part 2)?
                if (house < MAX_HOUSES2 && elf >= house/50)
                {
                    // Set presents for houses2
                    set_house2(house, 11 * elf);
                }
            }
        }
    }
}

// Deliver the present elf by elf
__global__ void deliver_by_elf()
{
    // Loop over elves (starting at 1)
    for(uint32_t elf = 1 + threadIdx.x + blockIdx.x * blockDim.x;
        elf < MAX_HOUSES1;
        elf += gridDim.x * blockDim.x)
    {
        // Presents for this elf (Part 1)
        for (uint32_t j = elf; j < MAX_HOUSES1; j += elf)
        {
            set_house1(j, 10 * elf);
        }
        // Presents for this elf (Part 2)
        for (uint32_t j = elf; j <= 50 * elf && j < MAX_HOUSES2; j += elf)
        {
            set_house2(j, 11 * elf);
        }
    }
}

// After the house are delivereed, find the smallest solution
__host__ uint32_t check_solution1()
{
    for (uint32_t house = 1; house < MAX_HOUSES1; ++house)
    {
        if (houses1[house] >= PUZZLE_INPUT)
        {
            return house;
        }
    }
    return ULONG_MAX;
}

// After the house are delivereed, find the smallest solution
__host__ uint32_t check_solution2()
{
    for (uint32_t house = 1; house < MAX_HOUSES2; ++house)
    {
        if (houses2[house] >= PUZZLE_INPUT)
        {
            return house;
        }
    }
    return ULONG_MAX;
}

void timing_code(uint32_t blocks, uint32_t threads, bool by_elf)
{
    // Initialize the houses to 0
    reset_houses<<<blocks,threads>>>();
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    if (by_elf)
    {
        deliver_by_elf<<<blocks,threads>>>();
        std::cout << "By elf: ";
    }
    else
    {
        deliver_by_house<<<blocks,threads>>>();
        std::cout << "By house: ";
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << blocks << " blocks, " << threads << " threads, ";
    std::cout << std::chrono::duration<double>(end - start).count() << " seconds\n";

    auto soln1 = check_solution1();
    assert(soln1 == 831600);
    auto soln2 = check_solution2();
    assert(soln2 == 884520);
    return;
}

int main()
{
    // Set up housing heaps
    cudaMallocManaged(&houses1, MAX_HOUSES1 * sizeof(uint32_t));
    cudaMallocManaged(&houses2, MAX_HOUSES2 * sizeof(uint32_t));

    // Timing for different numbers of blocks and threads
    std::vector<uint32_t> blocksValues = {32, 64, 128, 256, 512, 1024};
    std::vector<uint32_t> threadsValues = {128, 256};
    for (auto by_elf : {true, false})
    {
        for (auto blocks : blocksValues)
        {
            for(auto threads : threadsValues)
            {
                timing_code(blocks, threads, by_elf);                
            }
        }
    }

    cudaFree(houses1);
    cudaFree(houses2);
}

