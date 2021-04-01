#include <fstream>
#include <vector>
#include <string>

// Read the data from the input file into a vector of strings, one line per element
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

// Do a reduction to sum the elements created by each thread in a block 
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