#include <iostream>
#include "aoc_utils.hh"  // for data_lines and reduction2D

// Number of steps to take after intial state
const int STEPS = 100;

// Light grid size
const size_t SIZE = 100;
// Largest power of 2 bigger than SIZE (makes counting code easier)
const size_t THREADS = 128; 

// Light grid for Part 1
__device__ __managed__ int lights1[SIZE][SIZE];
// Light grid for Part 2
__device__ __managed__ int lights2[SIZE][SIZE];

/**
 * \brief Determine the next values for a particular light, given the values of its neighbors
 *
 * \param x First coordinate of light in grid
 * \param y Second coordinate of light in grid
 * \param part Part 1 or Part 2
 * \return Value at next step
 */
__device__ int next_value(int x, int y, int part)
{
    // Count of neighbors that are on
    int neighbors = 0;
    // Loop over the neighbors
    for (int i = x - 1; i <= x + 1; ++i)
    {
        for(int j = y - 1; j <= y + 1; ++j)
        {
            // Skip self
            if (i == x && j == y)
            {
                continue;
            }
            if (i >= 0 && i < SIZE && j >=0 && j < SIZE)
            {
                if (part == 1)
                    neighbors += lights1[i][j];
                else
                    neighbors += lights2[i][j];
            }
        }
    }

    // Return value
    if (part == 1)
        return ((neighbors == 3) || (neighbors == 2 && lights1[x][y] == 1));
    else
        return ((neighbors == 3) || (neighbors == 2 && lights2[x][y] == 1));
}

// Take one step for light states
__global__ void one_step()
{
    // Each thread is responsible for a row
    const int row = threadIdx.x;
    // New values for each column in this row (Part 1)
    int newValues1[SIZE];
    // New values for each column in this row (Part 2)
    int newValues2[SIZE];
    if (row < SIZE)
    {
        // Get next values
        for(int col = 0; col < 100; ++col)
        {
            newValues1[col] = next_value(row, col, 1);
            newValues2[col] = next_value(row, col, 2);
        }
    }

    // Sync to make sure not to overwrite values
    __syncthreads();

    if (row < SIZE)
    {
        for(int col = 0; col < 100; ++col)
        {
            lights1[row][col] = newValues1[col];
            lights2[row][col] = newValues2[col];
        }
        // Corners stay on
        if (row == 0 || row == 99)
        {
            lights2[row][0] = 1;
            lights2[row][99] = 1;
        }        
    }

    // Sync again before moving on to next step
    __syncthreads();
}

// Count the number of lights that are on
__global__ void count_lights(int* counts)
{
    // Shared counts for part 1
    __shared__ int64_t totalCount1[THREADS];
    // Shared counts for part 2
    __shared__ int64_t totalCount2[THREADS];
    // Thread count of lights for part 1
    int64_t count1 = 0;
    // Thread count of lights for part 1
    int64_t count2 = 0;
    // Alias for column
    int64_t col = threadIdx.x;
    // Count the lights
    if (col < SIZE)
    {
        for (int row = 0; row < SIZE; ++row)
        {
            count1 += lights1[row][col];
            count2 += lights2[row][col];
        }
    }

    // Set the counts to be visible to all threads in the block
    totalCount1[col] = count1;
    totalCount2[col] = count2;
    __syncthreads();

    // Perform reduction to get total count into column 0
    reduction(totalCount1, col);
    reduction(totalCount2, col);

    // Set output counts
    if (col== 0)
    {
        counts[0] = totalCount1[0];
        counts[1] = totalCount2[0];
    }
}

int main()
{
    // Read in the initial state of the lights
    auto dataLines = data_lines("../data/day18.input.txt");

    // Set the initial state of the lights
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            lights1[i][j] = (dataLines[i][j] == '#') ? 1 : 0;
            lights2[i][j] = (dataLines[i][j] == '#') ? 1 : 0;
        }
    }
    lights2[0][0] = 1;
    lights2[0][99] = 1;
    lights2[99][0] = 1;
    lights2[99][99] = 1;

    // Loop over the steps and "animate"
    for (int step = 0; step < STEPS; ++step)
    {
        one_step<<<1,THREADS>>>();
        cudaDeviceSynchronize();
    }

    // Count the lights
    int counts[2] = {0, 0};
    int* dev_counts;
    cudaMalloc(&dev_counts, 2 * sizeof(int));
    count_lights<<<1,THREADS>>>(dev_counts);
    cudaDeviceSynchronize();

    cudaMemcpy(&counts, dev_counts, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Solution 1: " << counts[0] << std::endl;
    std::cout << "Solution 2: " << counts[1] << std::endl;

    cudaFree(dev_counts);
}
