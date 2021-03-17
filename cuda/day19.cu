#include <iostream>
#include <string>
#include <vector>

#include "aoc_utils.hh"
#include "Transforms.hh"
#include "Molecule.hh"

const size_t MAX_CHILD_NUM = 60;

const size_t THREADS = 64;
const size_t BLOCKS = 128;
const size_t MAX_HEAP_SIZE = THREADS * BLOCKS;

__device__ __managed__ Molecule* heap;
__device__ __managed__ size_t heapSize = 0;

__device__ __managed__ size_t countMols = 0;
__device__ __managed__ size_t bestSteps = INT_MAX;

__global__ void initialize_heap(Molecule* start)
{
    for (int64_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
        idx < MAX_HEAP_SIZE; 
        idx += blockDim.x * gridDim.x)
    {
        if (idx == 0)
        {
            heap[idx] = start[0];
            heapSize = 1;
        }
        else
        {
            heap[idx] = Molecule();
        }
    }
}

__global__ void fabricate_molecules(Molecule* start)
{
    for (int64_t transformIdx = threadIdx.x; transformIdx < TSIZE; transformIdx += blockDim.x)
    {
        for (int64_t childNum = 0; childNum < MAX_CHILD_NUM; ++childNum)
        {
            int64_t heapIdx = transformIdx * MAX_CHILD_NUM + childNum;
            Molecule m = start[0].fabricate(transformIdx, childNum);
            heap[heapIdx] = m;
        }
        atomicAdd(&heapSize, MAX_CHILD_NUM);
    }
}

__global__ void count_unique(int64_t* totalCount)
{
    __shared__ int64_t count[THREADS];
    int64_t uniqueCount = 0;
    for (int64_t heapIdx1 = threadIdx.x; heapIdx1 < heapSize; heapIdx1 += blockDim.x)
    {
        if (heap[heapIdx1].steps == INT_MAX)
        {
            continue;
        }
        bool is_duplicate = false;
        for (int64_t heapIdx2 = 0; heapIdx2 < heapIdx1; ++heapIdx2)
        {
            if (heap[heapIdx1] == heap[heapIdx2])
            {
                is_duplicate = true;
                break;
            }
        }
        if (!is_duplicate)
        {
            ++uniqueCount;
        }
    }
    count[threadIdx.x] = uniqueCount;
    __syncthreads();
    reduction(count, threadIdx.x);
    if (threadIdx.x == 0)
    {
        totalCount[0] = count[0];
    }
}

__global__ void deconstruct_molecules()
{
    int64_t reserved_size = heapSize + TSIZE * blockIdx.x + TSIZE;
    if (reserved_size < MAX_HEAP_SIZE)
    {
        Molecule start = heap[blockIdx.x];
        for(int64_t transformIdx = threadIdx.x; transformIdx < TSIZE; transformIdx += blockDim.x)
        {
            Molecule m = start.deconstruct(transformIdx);
            if (m.steps < bestSteps)
            {
                if (m.msize == 1)
                {
                    if (m.molecule[0] == 16)
                    {
                        atomicMin(&bestSteps, m.steps);
                    }
                }
                else
                {
                    int64_t newIdx = heapSize + TSIZE * blockIdx.x + transformIdx;
                    heap[newIdx] = m;
                }
            }
            if (threadIdx.x == 0)
            {
                heap[blockIdx.x] = Molecule();
            }
        }
    }
}

__global__ void bitonic_sort_step(int64_t j, int64_t k)
{
    int64_t i = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t ij = i ^ j;
    if (ij > i)
    {
        int64_t ik = i&k;
        if ( (ik == 0 && heap[ij] < heap[i]) ||
             (ik != 0 && heap[i] < heap[ij]) )
        {
            Molecule temp = heap[i];
            heap[i] = heap[ij];
            heap[ij] = temp;
        }
    }
}

__host__ void bitonic_sort()
{
    for (int64_t k = 2; k <= MAX_HEAP_SIZE; k <<= 1)
    {
        for(int64_t j = k >> 1; j > 0; j = j >> 1)
        {
            bitonic_sort_step<<<BLOCKS,THREADS>>>(j, k);
        }
    }
    cudaDeviceSynchronize();
}

__global__ void set_heap_size()
{
    heapSize = 0;
    for (size_t i = 0; i < MAX_HEAP_SIZE; ++i)
    {
        if (heap[i].steps == INT_MAX)
        {
            heapSize = i;
            break;
        }
    }
}

int main()
{
    size_t molSize = sizeof(Molecule);
    cudaMallocManaged(&heap, MAX_HEAP_SIZE * molSize);

    Molecule heap_start(to_int64(MOLECULE_STRING));
    Molecule* dev_heap_start;
    cudaMalloc(&dev_heap_start, molSize);
    cudaMemcpy(dev_heap_start, &heap_start, molSize, cudaMemcpyHostToDevice);


    fabricate_molecules<<<1,THREADS>>>(dev_heap_start);
    cudaDeviceSynchronize();

    int64_t totalCount = 0;
    int64_t* dev_total_count;
    cudaMalloc(&dev_total_count, sizeof(int64_t));
    count_unique<<<1,THREADS>>>(dev_total_count);
    cudaMemcpy(&totalCount, dev_total_count, sizeof(int64_t), cudaMemcpyDeviceToHost);

    std::cout << "Solution 1: " << totalCount << std::endl;


    initialize_heap<<<BLOCKS,THREADS>>>(dev_heap_start);
    cudaDeviceSynchronize();

    while (heapSize > 0 && (heapSize + TSIZE < MAX_HEAP_SIZE) && bestSteps == INT_MAX)
    {
        deconstruct_molecules<<<1,THREADS>>>();
        cudaDeviceSynchronize();
        bitonic_sort();
        set_heap_size<<<1,1>>>();
        cudaDeviceSynchronize();
    }
    std::cout << "Solution 2: " << bestSteps << std::endl;

    cudaFree(heap);
    cudaFree(dev_heap_start);
    cudaFree(dev_total_count);
}