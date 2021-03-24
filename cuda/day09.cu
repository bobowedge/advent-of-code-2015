#include <cstdint>
#include <iostream>
#include "Route.hh" // Route class

// Somewhat arbitrary choice of threads and blocks
const size_t THREADS = 64;
const size_t BLOCKS = 64;

// Minimum route distance (part 1)
__device__ __managed__ int minDistance = INT_MAX;
// Maximum route distance (part 2)
__device__ __managed__ int maxDistance = 0;

__global__ void find_min_max_routes()
{
    int stop = powf(DESTINATIONS, DESTINATIONS);
    for (int rid = threadIdx.x + blockIdx.x * blockDim.x; rid < stop; rid += blockDim.x * gridDim.x)
    {
        // Enumerate route
        Route route(rid);
        // Check if route visit each location exacly once
        if (route.valid())
        {
            auto distance = route.distance();
            atomicMin(&minDistance, distance);
            atomicMax(&maxDistance, distance);
        }
    }
}

int main()
{
    find_min_max_routes<<<BLOCKS, THREADS>>>();
    cudaDeviceSynchronize();
    std::cout << "Solution 1: " << minDistance << std::endl;
    std::cout << "Solution 2: " << maxDistance << std::endl;
}