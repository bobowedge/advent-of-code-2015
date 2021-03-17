#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include "aoc_utils.hh"

const dim3 BLOCKS(16, 16, 1);
const dim3 THREADS(64, 1, 1);

__device__ __managed__ int lights1[1000][1000];
__device__ __managed__ int lights2[1000][1000];
__device__ __managed__ int numLightsOn = 0;
__device__ __managed__ int brightness = 0;

int get_type(const std::string& line)
{
    if (line.substr(0, 6) == std::string("toggle"))
    {
        return -1;    
    }
    if (line.substr(0, 7) == std::string("turn on"))
    {
        return 1;
    }
    return 0;
}

int4 parse_coordinates(const std::string& line, int type)
{
    int skip = 0;
    switch(type)
    {
        case 0:
        {
            skip = 9;
            break;
        }
        case 1:
        {
            skip = 8;
            break;
        }
        case -1:
        {
            skip = 7;
            break;
        }
        default:
            break;
    }

    int4 data = make_int4(0, 0, 0, 0);
    auto idx0 = line.find(",", skip);
    data.x = std::stol(line.substr(skip, idx0 - skip));
    auto idx1 = line.find(" ", idx0);
    ++idx0;
    data.y = std::stol(line.substr(idx0, idx1 - idx0));
    idx1 += std::string(" through ").size();
    auto idx2 = line.find(",", idx1);
    data.z = std::stol(line.substr(idx1, idx2 - idx1));
    ++idx2;
    data.w = std::stol(line.substr(idx2, line.size() - idx2));
    return data;
}

__global__ void apply_pattern(int4* coordinates, int* type)
{
    const int &row0 = coordinates[0].x;
    const int &row1 = coordinates[0].z;
    const int &col0 = coordinates[0].y;
    const int &col1 = coordinates[0].w;
    for(int row = blockIdx.x; row < 1000; row += gridDim.y)
    {
        if (row0 > row || row > row1)
        {
            continue;
        }
        for(int col = blockIdx.y + threadIdx.x * gridDim.x; 
            col < 1000; 
            col += gridDim.x * blockDim.x)
        {
            if (col0 > col || col > col1)
            {
                continue;
            }
            if (type[0] == -1)
            {
                lights1[row][col] ^= 1;
                lights2[row][col] += 2;
            }
            else if (type[0] == 0)
            {
                lights1[row][col] = 0;
                if (lights2[row][col] > 0)
                {
                    --lights2[row][col];
                }
            }
            else
            {
                lights1[row][col] = 1;
                ++lights2[row][col];
            }
        }
    }
}

__global__ void reset_lights()
{
    for (int row = blockIdx.x; row < 1000; row += gridDim.y)
    {
        for(int col = blockIdx.y + threadIdx.x * gridDim.x; 
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

__global__ void count_lights()
{
    int count = 0;
    int bright = 0;
    for (int row = blockIdx.x; row < 1000; row += gridDim.y)
    {
        for(int col = blockIdx.y + threadIdx.x * gridDim.x; 
            col < 1000; 
            col += gridDim.x * blockDim.x)
        {
            if (lights1[row][col] == 1)
            {
                ++count;
            }
            bright += lights2[row][col];
        }
    }
    atomicAdd(&numLightsOn, count);
    atomicAdd(&brightness, bright);
}

std::pair<int, int> solution()
{
    int type = 0;
    int4 coordinates = make_int4(0, 0, 0, 0);
    int* dev_type;
    int4* dev_coordinates;

    cudaMalloc(&dev_type, sizeof(int));
    cudaMalloc(&dev_coordinates, sizeof(int4));
    
    reset_lights<<<BLOCKS,THREADS>>>();

    auto dataLines = data_lines("../data/day06.input.txt");
    for (auto line : dataLines)
    {
        type = get_type(line);
        coordinates = parse_coordinates(line, type);
        cudaMemcpy(dev_type, &type, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_coordinates, &coordinates, sizeof(int4), cudaMemcpyHostToDevice);
        apply_pattern<<<BLOCKS,THREADS>>>(dev_coordinates, dev_type);
    }
    count_lights<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();
    cudaFree(dev_coordinates);
    cudaFree(dev_type);
    return std::make_pair(numLightsOn, brightness);
}

int main()
{
    auto sol = solution(); 
    std::cout << "Solution 1: " << sol.first << std::endl;
    std::cout << "Solution 2: " << sol.second << std::endl;
}