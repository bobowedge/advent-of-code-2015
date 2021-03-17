#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "aoc_utils.hh"

const size_t THREADS = 32;
const size_t BLOCKS = 32;

__device__ int64_t stdmin(int64_t a, int64_t b, int64_t c)
{
    if (a < b && a < c)
    {
        return a;
    }
    if (b < c)
    {
        return b;
    }
    return c;
}

__global__ void count_paper(int64_t *dimensions, int64_t *papers, int64_t *ribbons, const size_t N)
{
    __shared__ int64_t cache1[THREADS];
    __shared__ int64_t cache2[THREADS];
    int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int64_t cacheIndex = threadIdx.x;

    int64_t tempPaper = 0;
    int64_t tempRibbon = 0;
    while (tid < N)
    {
        // 3 values for dimensions
        int64_t length = dimensions[3 * tid];
        int64_t width = dimensions[3 * tid + 1];
        int64_t height = dimensions[3 * tid + 2];
        int64_t extraPaper = stdmin(length*width, width*height, length*height);
        int64_t extraRibbon = stdmin(length + width, width + height, length + height);
        tempPaper += 2 * (length * width + width * height + length * height) + extraPaper;
        tempRibbon += 2 * extraRibbon + length * width * height;
        tid += 3 * blockDim.x * gridDim.x;
    }
    
    cache1[cacheIndex] = tempPaper;
    cache2[cacheIndex] = tempRibbon;
    
    __syncthreads();

    reduction(cache1, cacheIndex);
    reduction(cache2, cacheIndex);

    if (cacheIndex == 0)
    {
        papers[blockIdx.x] = cache1[0];
        ribbons[blockIdx.x] = cache2[0];
    }
}


int main() 
{
    auto dataLines = data_lines("../data/day02.input.txt");

    std::vector<int64_t> dimensions;
    dimensions.reserve(3 * dataLines.size());

    /// Convert line to integers
    for (auto line : dataLines)
    {
        size_t lineSize = line.size();
        std::string value = "";
        for (auto c : line)
        {
            if (c != 'x')
            {
                value += std::string(1, c);
            }
            else
            {
                dimensions.push_back(std::stol(value));
                value = "";
            }
        }
        dimensions.push_back(std::stol(value));
    }

    int64_t* dev_dimensions;
    int64_t* dev_partial_papers;
    int64_t* dev_partial_ribbons;

    size_t dev_dim_size = dimensions.size() * sizeof(int64_t);
    size_t dev_pp_size = BLOCKS * sizeof(int64_t);
    size_t dev_rb_size = BLOCKS * sizeof(int64_t);

    cudaMalloc((void **)&dev_dimensions, dev_dim_size);
    cudaMalloc((void **)&dev_partial_papers, dev_pp_size);
    cudaMalloc((void **)&dev_partial_ribbons, dev_rb_size);
    cudaMemcpy(dev_dimensions, dimensions.data(), dev_dim_size, cudaMemcpyHostToDevice);

    count_paper<<<BLOCKS,THREADS>>>(dev_dimensions, 
        dev_partial_papers, dev_partial_ribbons, dimensions.size());

    std::vector<int64_t> partial_papers(BLOCKS);
    std::vector<int64_t> partial_ribbons(BLOCKS);
    cudaMemcpy(partial_papers.data(), dev_partial_papers, dev_pp_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(partial_ribbons.data(), dev_partial_ribbons, dev_rb_size, cudaMemcpyDeviceToHost);

    int64_t papers = std::accumulate(partial_papers.begin(), 
                                     partial_papers.end(), 
                                     static_cast<int64_t>(0));

    int64_t ribbons = std::accumulate(partial_ribbons.begin(), 
                                     partial_ribbons.end(), 
                                     static_cast<int64_t>(0));
    
    std::cout << "Solution 1: " << papers << std::endl;
    std::cout << "Solution 2: " << ribbons << std::endl;
    
    cudaFree (dev_dimensions);
    cudaFree (dev_partial_papers);
    cudaFree (dev_partial_ribbons);
}
