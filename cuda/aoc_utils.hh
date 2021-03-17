#include <fstream>
#include <vector>
#include <string>

std::vector<std::string> data_lines(const std::string& filename)
{
    std::vector<std::string> dataLines;
    std::ifstream data(filename);
    std::string line;
    while(!data.eof())
    {
        std::getline(data, line);
        dataLines.push_back(line);
    }
    return dataLines;
}

__device__ void reduction(int64_t *cache, int64_t cacheIndex)
{
    int64_t index = blockDim.x >> 1;
    while (index > 0)
    {
        if (cacheIndex < index)
        {
            cache[cacheIndex] += cache[cacheIndex + index];
        }
        __syncthreads();
        index >>= 1;
    }
    return;
}

__device__ void reduction2D(int *cache, int64_t cacheIndex)
{
    int64_t index = (blockDim.x * blockDim.y) >> 1;
    while (index > 0)
    {
        if (cacheIndex < index)
        {
            cache[cacheIndex] += cache[cacheIndex + index];
        }
        __syncthreads();
        index >>= 1;
    }
    return;
}