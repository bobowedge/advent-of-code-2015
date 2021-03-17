#include <cstdint>
#include <iostream>

const size_t THREADS = 64;
const size_t BLOCKS = 64;

__device__ __managed__ int minDistance = INT_MAX;
__device__ __managed__ int maxDistance = 0;

const int DESTINATIONS = 8;

__device__ int leg_distance(int src, int dest)
{
    if (src == dest || src < 0 || src > 7 || dest < 0 || dest > 7)
    {
        return -1;
    }

    if (src > dest)
    {
        int tmp = src;
        src = dest;
        dest = tmp;
    }

    switch(src)
    {
        case 0:
        {
            switch(dest)
            {
                case 1:
                    return 66;
                case 2:
                    return 28;
                case 3:
                    return 60;
                case 4:
                    return 34;
                case 5:
                    return 34;
                case 6:
                    return 3;
                case 7:
                    return 108;
            }
        }
        case 1:
        {
            switch(dest)
            {
                case 2:
                    return 22;
                case 3:
                    return 12;
                case 4:
                    return 91;
                case 5:
                    return 121;
                case 6:
                    return 111;
                case 7:
                    return 71;
            }
        }
        case 2:
        {
            switch(dest)
            {
                case 3:
                    return 39;
                case 4:
                    return 113;
                case 5:
                    return 130;
                case 6:
                    return 35;
                case 7:
                    return 40;
            }
        }
        case 3:
        {
            switch(dest)
            {
                case 4:
                    return 63;
                case 5:
                    return 21;
                case 6:
                    return 57;
                case 7:
                    return 83;
            }
        }
        case 4:
        {
            switch(dest)
            {
                case 5:
                    return 9;
                case 6:
                    return 50;
                case 7:
                    return 60;
            }
        }
        case 5:
        {
            switch(dest)
            {
                case 6:
                    return 27;
                case 7:
                    return 81;
            }
        }
        case 6:
        {
            return 90;
        }
    }
    return -1;
}

class Route
{
public:
    __device__ Route(int N)
    {
        for (int i = 0; i < DESTINATIONS; ++i)
        {
            route_[i] = N % DESTINATIONS;
            N /= DESTINATIONS;
        }
    }

    __device__ bool valid() const
    {
        for (int i = 0; i < DESTINATIONS; ++i)
        {
            for (int j = i + 1; j < DESTINATIONS; ++j)
            {
                if (route_[i] == route_[j])
                {
                    return false;
                }
            }
            if (route_[i] < 0 || route_[i] >= DESTINATIONS)
            {
                return false;
            }
        }
        return true;
    }

    __device__ int distance() const
    {
        int distance = 0;
        for (int i = 0; i < DESTINATIONS - 1; ++i)
        {
            distance += leg_distance(route_[i], route_[i+1]);
        }
        return distance;
    }
private:
    int route_[DESTINATIONS];
};

__global__ void find_min_max()
{
    int start = threadIdx.x + blockIdx.x * blockDim.x;
    auto stop = __powf(DESTINATIONS, DESTINATIONS);
    int increment = blockDim.x * gridDim.x;
    for (int tid = start; tid < stop; tid += increment)
    {
        Route route(tid);
        if (route.valid())
        {
            auto distance = route.distance();
            atomicMin(&minDistance, distance);
            atomicMax(&maxDistance, distance);
        }
    }
}

__global__ void reset_values()
{
    minDistance = INT_MAX;
    maxDistance = 0;
}

int main()
{
    reset_values<<<1,1>>>();
    cudaDeviceSynchronize();
    find_min_max<<<BLOCKS, THREADS>>>();
    cudaDeviceSynchronize();

    std::cout << "Solution 1: " << minDistance << std::endl;
    std::cout << "Solution 2: " << maxDistance << std::endl;
}