// nvcc -g -o day02.exe day02.cu
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include "aoc_utils.hh" // for reduction and data_lines

// Somewhat arbitrary values for THREADS and BLOCKS
const size_t THREADS = 32;
const size_t BLOCKS = 32;

// Find the minimum of three values
__device__ int64_t min3(int64_t a, int64_t b, int64_t c)
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

/**
 * \brief Count the amount of wrapping paper and ribbons needed
 * 
 * \param dimensions Dimensions of each box
 * \param papers Wrapping paper counted by each block
 * \param ribbons Ribbons counted by each block
 * \param N Number of boxes in dimensions
 */
__global__ void count_paper_and_ribbon(int64_t *dimensions, int64_t *papers, int64_t *ribbons, const size_t N)
{
    // Paper for this thread (Part 1)
    int64_t sumPaper = 0;
    // Ribbons for this thread (Part 2)
    int64_t sumRibbon = 0;
    // Loop over some boxes (different ones for each thread)
    for (int64_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N; tid += 3 * blockDim.x * gridDim.x)
    {
        // 3 values for dimensions
        int64_t length = dimensions[3 * tid];
        int64_t width = dimensions[3 * tid + 1];
        int64_t height = dimensions[3 * tid + 2];

        // Paper needed for Part 1 : surface area + smallest side
        sumPaper += 2 * (length * width + width * height + length * height);
        sumPaper += min3(length*width, width*height, length*height);
        
        // Ribbon needed for Part 2 : smallest perimeter + volume
        sumRibbon += 2 * min3(length + width, width + height, length + height);
        sumRibbon += length * width * height;
    }

    // Block shared memory array
    __shared__ int64_t paperCache[THREADS];
    __shared__ int64_t ribbonCache[THREADS];
    const int64_t cacheIndex = threadIdx.x;

    paperCache[cacheIndex] = sumPaper;
    ribbonCache[cacheIndex] = sumRibbon;
    
    __syncthreads();

    reduction(paperCache, cacheIndex);
    reduction(ribbonCache, cacheIndex);

    if (cacheIndex == 0)
    {
        papers[blockIdx.x] = paperCache[0];
        ribbons[blockIdx.x] = ribbonCache[0];
    }
}

int main() 
{
    // Read the data into a vector
    auto dataLines = data_lines("../data/day02.input.txt");

    std::vector<int64_t> dimensions;
    dimensions.reserve(3 * dataLines.size());

    /// Convert each line in dataLines to integer dimensions (3 per box)
    for (auto line : dataLines)
    {
        size_t lineSize = line.size();
        std::string value = "";
        for (auto c : line)
        {
            // Each line is LxWxH for some integers L,W,H
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

    // Device input dimensions (3 per box)
    int64_t* dev_dimensions;
    size_t dev_dim_size = dimensions.size() * sizeof(int64_t);
    cudaMalloc((void **)&dev_dimensions, dev_dim_size);

    // Device output paper needed for each block (Part 1)
    int64_t* dev_partial_papers;
    size_t dev_pp_size = BLOCKS * sizeof(int64_t);
    cudaMalloc((void **)&dev_partial_papers, dev_pp_size);

    // Device output ribbon needed for each block (Part 2)
    int64_t* dev_partial_ribbons;
    size_t dev_rb_size = BLOCKS * sizeof(int64_t);
    cudaMalloc((void **)&dev_partial_ribbons, dev_rb_size);

    // Copy dimensions to device
    cudaMemcpy(dev_dimensions, dimensions.data(), dev_dim_size, cudaMemcpyHostToDevice);

    // Calculate paper and ribbon
    count_paper_and_ribbon<<<BLOCKS,THREADS>>>(dev_dimensions, 
        dev_partial_papers, dev_partial_ribbons, dimensions.size());

    // Copy paper data back from device
    std::vector<int64_t> partial_papers(BLOCKS);
    cudaMemcpy(partial_papers.data(), dev_partial_papers, dev_pp_size, cudaMemcpyDeviceToHost);
    // Sum paper data from each block
    int64_t papers = std::accumulate(partial_papers.begin(), 
                                     partial_papers.end(), 
                                     static_cast<int64_t>(0));
    // Print solution for Part 1
    std::cout << "Solution 1: " << papers << std::endl;
    
    // Copy ribbon data back from device
    std::vector<int64_t> partial_ribbons(BLOCKS);
    cudaMemcpy(partial_ribbons.data(), dev_partial_ribbons, dev_rb_size, cudaMemcpyDeviceToHost);
    // Sum ribbon data from each block
    int64_t ribbons = std::accumulate(partial_ribbons.begin(), 
                                     partial_ribbons.end(), 
                                     static_cast<int64_t>(0));   
    // Print solution for Part 2
    std::cout << "Solution 2: " << ribbons << std::endl;
    
    // Free device memory
    cudaFree (dev_dimensions);
    cudaFree (dev_partial_papers);
    cudaFree (dev_partial_ribbons);
}
