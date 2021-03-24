#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "aoc_utils.hh"   // For data lines

// Block dimensions
const dim3 BLOCKS(8, 16, 1);
// Thread dimensions
const dim3 THREADS(16, 32, 1);

// Light grid for part 1
__device__ __managed__ int lights1[1000][1000];
// Light grid for part 2
__device__ __managed__ int lights2[1000][1000];
// Number of lights on (part 1)
__device__ __managed__ int numLightsOn = 0;
// Brightness (part 2)
__device__ __managed__ int brightness = 0;

// Parse the input line to get the coordinate grid and type
std::pair<int,int4> parse_line(const std::string line)
{
    int skip = 0;
    int type = 0;
    int4 grid = make_int4(0, 0, 0, 0);
    if (line.substr(0, 6) == std::string("toggle"))
    {
        type = 0;
        skip = 7;
    }
    else if (line.substr(0, 7) == std::string("turn on"))
    {
        type = 1;
        skip = 8;
    }
    else
    {
        type = -1;
        skip = 9;
    }
    auto idx0 = line.find(",", skip);
    grid.x = std::stol(line.substr(skip, idx0 - skip));
    auto idx1 = line.find(" ", idx0);
    ++idx0;
    grid.y = std::stol(line.substr(idx0, idx1 - idx0));
    idx1 += std::string(" through ").size();
    auto idx2 = line.find(",", idx1);
    grid.z = std::stol(line.substr(idx1, idx2 - idx1));
    ++idx2;
    grid.w = std::stol(line.substr(idx2, line.size() - idx2));
    return std::make_pair(type, grid);
}

/**
 *  \brief Apply a single grid pattern 
 * 
 * \param grid Four integers representing grid to adjust:
 *              (first row, first col, last row, last col)
 * \param type Instruction type (turn off = -1, turn on = 1, toggle = 0)
 */
__global__ void apply_instruction(int type, int4 grid)
{
    const int firstRow = grid.x;
    const int lastRow = grid.z;
    const int firstColumn = grid.y;
    const int lastColumn = grid.w;
    for(int row = threadIdx.y + blockIdx.y * blockDim.y; 
            row < 1000; 
            row += gridDim.y * blockDim.y)
    {
        // Check if this row is in the instruction
        if (row < firstRow || row > lastRow)
        {
            continue;
        }
        for(int column = threadIdx.x + blockIdx.x * blockDim.x; 
            column < 1000; 
            column += gridDim.x * blockDim.x)
        {
            // Check if this column is in the instruction
            if (column < firstColumn || column > lastColumn)
            {
                continue;
            }
            //// Apply instruction
            // Toggle
            if (type == 0)
            {
                lights1[row][column] ^= 1;
                lights2[row][column] += 2;
            }
            // Turn off
            else if (type == -1)
            {
                lights1[row][column] = 0;
                if (lights2[row][column] > 0)
                {
                    --lights2[row][column];
                }
            }
            // Turn on
            else
            {
                lights1[row][column] = 1;
                ++lights2[row][column];
            }
        }
    }
}

// Reset the lights grids, number of lights on and brightness to 0
__global__ void reset_lights()
{
    for(int row = threadIdx.y + blockIdx.y * blockDim.y; 
        row < 1000; 
        row += gridDim.y * blockDim.y)
    {
        for(int col = threadIdx.x + blockIdx.x * blockDim.x; 
            col < 1000; 
            col += gridDim.x * blockDim.x)
        {
            lights1[row][col] = 0;
            lights2[row][col] = 0;
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    {
        numLightsOn = 0;
        brightness = 0;
    }
}

// Count the number of lights on and the brightness
__global__ void count_lights()
{
    // Count and brightness for this thread
    int count = 0;
    int bright = 0;
    for(int row = threadIdx.y + blockIdx.y * blockDim.y; 
        row < 1000; 
        row += gridDim.y * blockDim.y)
    {
        for(int col = threadIdx.x + blockIdx.x * blockDim.x; 
            col < 1000; 
            col += gridDim.x * blockDim.x)
        {
            count += lights1[row][col];
            bright += lights2[row][col];
        }
    }
    // Add to global counts 
    atomicAdd(&numLightsOn, count);
    atomicAdd(&brightness, bright);
}

std::pair<int, int> solution()
{
    // Reset the lights to 0
    reset_lights<<<BLOCKS,THREADS>>>();

    // Read the data
    auto dataLines = data_lines("../data/day06.input.txt");
    // Loop over the instructions
    for (auto line : dataLines)
    {
        // Parse the instruction from the line
        auto typeGridPair = parse_line(line);
        int type = typeGridPair.first;
        int4 grid = typeGridPair.second;
        // Apply the instruction on the device
        apply_instruction<<<BLOCKS,THREADS>>>(type, grid);
    }
    cudaDeviceSynchronize();
    
    // Count the lights that are on and the brightness
    count_lights<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();
    
    return std::make_pair(numLightsOn, brightness);
}

int main()
{
    auto sol = solution(); 
    std::cout << "Solution 1: " << sol.first << std::endl;
    std::cout << "Solution 2: " << sol.second << std::endl;
}