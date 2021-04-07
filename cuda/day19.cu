#include <iostream>
#include <string>
#include <vector>

#include "aoc_utils.hh"
#include "Molecule.hh"

// Given input string
const std::string MOLECULE_STRING = "CRnCaSiRnBSiRnFArTiBPTiTiBFArPBCaSiThSiRnTiBPBPMgArCaSiRnTiMgArCaSiThCaSiRnFArRnSiRnFArTiTiBFArCaCaSiRnSiThCaCaSiRnMgArFYSiRnFYCaFArSiThCaSiThPBPTiMgArCaPRnSiAlArPBCaCaSiRnFYSiThCaRnFArArCaCaSiRnPBSiRnFArMgYCaCaCaCaSiThCaCaSiAlArCaCaSiRnPBSiAlArBCaCaCaCaSiThCaPBSiThPBPBCaSiRnFYFArSiThCaSiRnFArBCaCaSiRnFYFArSiThCaPBSiThCaSiRnPMgArRnFArPTiBCaPRnFArCaCaCaCaSiRnCaCaSiRnFYFArFArBCaSiThFArThSiThSiRnTiRnPMgArFArCaSiThCaPBCaSiRnBFArCaCaPRnCaCaPMgArSiRnFYFArCaSiThRnPBPMgAr";

// Memory size of Molecule
const size_t MOL_SIZE = sizeof(Molecule);

// I wanted the largest heap size I could get without crashing the device in Part 2
// These values seem to work well for that
const size_t THREADS = 64;
const size_t BLOCKS = 128;
const size_t MAX_HEAP_SIZE = THREADS * BLOCKS;

// List of current molecules
__device__ __managed__ Molecule* heap;
// Current size of heap
__device__ __managed__ size_t heapSize = 0;

// Best value of steps to completely deconstruct input molecule (Part 2)
__device__ __managed__ size_t bestSteps = INT_MAX;

// Initialize the heap with the input value (Part 2)
__global__ void initialize_heap(Molecule* start)
{
    for (int64_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
        idx < MAX_HEAP_SIZE; 
        idx += blockDim.x * gridDim.x)
    {
        // First value is input molecule
        if (idx == 0)
        {
            heap[idx] = start[0];
            heapSize = 1;
        }
        // Everything else is invalid
        else
        {
            heap[idx] = Molecule();
        }
    }
}

// Make the possible children of `start` and add them to `heap`
__global__ void fabricate_molecules(Molecule* start)
{
    for (int64_t transformIdx = threadIdx.x; 
                 transformIdx < TSIZE; 
                 transformIdx += blockDim.x)
    {
        int64_t childNum = 0;
        // Produce first child
        Molecule m = start[0].fabricate(transformIdx, childNum);
        // Continue we produce an invalid child or run out of space
        while(m.is_valid() && heapSize < MAX_HEAP_SIZE)
        {
            // Add to heap
            int64_t heapIdx = atomicAdd(&heapSize, static_cast<size_t>(1));
            heap[heapIdx] = m;
            ++childNum;

            // Produce next child
            m = start[0].fabricate(transformIdx, childNum);
        }
    }
}

// Count the number of unique molecules on the heap
__global__ void count_unique(int64_t* totalCount)
{
    // Shared unique counts for the block
    __shared__ int64_t count[THREADS];
    int64_t uniqueCount = 0;
    for (int64_t heapIdx1 = threadIdx.x; heapIdx1 < heapSize; heapIdx1 += blockDim.x)
    {
        // Invalid molecule
        if (!heap[heapIdx1].is_valid())
        {
            continue;
        }
        // Check for prior duplicate
        bool isDuplicate = false;
        for (int64_t heapIdx2 = 0; heapIdx2 < heapIdx1; ++heapIdx2)
        {
            if (heap[heapIdx1] == heap[heapIdx2])
            {
                // Duplicate found
                isDuplicate = true;
                break;
            }
        }
        // Check if unique
        if (!isDuplicate)
        {
            ++uniqueCount;
        }
    }

    // Total counts across all threads
    count[threadIdx.x] = uniqueCount;
    __syncthreads();
    reduction(count, threadIdx.x);

    if (threadIdx.x == 0)
    {
        totalCount[0] = count[0];
    }
}

// Apply the reverse transformations to the top of the heap
__global__ void deconstruct_molecules()
{
    if (blockIdx.x < heapSize)
    {
        // Each block grabs a different Molecule from the top of the heap
        Molecule start = heap[blockIdx.x];
        // Each thread applies a different reverse transformation
        for(int64_t transformIdx = threadIdx.x; transformIdx < TSIZE; transformIdx += blockDim.x)
        {
            // Create the new molecule
            Molecule m = start.deconstruct(transformIdx);
            if (!m.is_valid())
            {
                continue;
            }
            // Reached 'e'
            if (m.msize == 1 && m.molecule[0] == 16)
            {
                atomicMin(&bestSteps, m.steps);
            }
            else if (m.steps < bestSteps)
            {
                // Add new molecule to the end of the heap
                int64_t heapIdx = atomicAdd(&heapSize, static_cast<size_t>(1));
                heap[heapIdx] = m;
            }
        }
        // This Molecule has been processed, so make it invalid
        if (threadIdx.x == 0)
        {
            heap[blockIdx.x] = Molecule();
        }
    }
}

// Single step of the bitonic_sort()
__global__ void bitonic_sort_step(int64_t j, int64_t k)
{
    // Each thread handles a different element
    int64_t i = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t ij = i ^ j;
    if (ij > i)
    {
        // Determines whether or not to swap indices
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

// Parallel sort algorithm
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

// Reset the heap size
__global__ void reset_heap_size()
{
    heapSize = 0;
    for (size_t i = 0; i < MAX_HEAP_SIZE; ++i)
    {
        // First invalid is heapSize (since heap is sorted)
        if (!heap[i].is_valid())
        {
            heapSize = i;
            break;
        }
    }
}

// Find the solution to part 1
int64_t solution1(const Molecule& input)
{
    // Copy input molecule to device
    Molecule* dev_input;
    cudaMalloc(&dev_input, MOL_SIZE);
    cudaMemcpy(dev_input, &input, MOL_SIZE, cudaMemcpyHostToDevice);

    // Generate all transformations
    fabricate_molecules<<<1,THREADS>>>(dev_input);
    cudaDeviceSynchronize();

    int64_t* dev_total_count;
    cudaMalloc(&dev_total_count, sizeof(int64_t));
    
    // Count the unique Molecules
    count_unique<<<1,THREADS>>>(dev_total_count);
    
    int64_t totalCount = 0;
    cudaMemcpy(&totalCount, dev_total_count, sizeof(int64_t), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_input);
    cudaFree(dev_total_count);

    return totalCount;
}

// Find a solution to part 2
void solution2(const Molecule& input)
{
    // Copy input molecule to device
    Molecule* dev_input;
    cudaMalloc(&dev_input, MOL_SIZE);
    cudaMemcpy(dev_input, &input, MOL_SIZE, cudaMemcpyHostToDevice);

    // Reset heap to just have input molecule
    initialize_heap<<<BLOCKS,THREADS>>>(dev_input);
    cudaDeviceSynchronize();
    
    // Continue until a path is found or heap gets too large
    while (heapSize > 0 && (heapSize + TSIZE < MAX_HEAP_SIZE) && bestSteps == INT_MAX)
    {
        // Apply each reverse transformation to the top of the heap
        // Any bigger than 3 blocks and the heap gets too big (1 seems fastest)
        deconstruct_molecules<<<1,THREADS>>>();
        cudaDeviceSynchronize();
        // Parallel sort of the heap
        bitonic_sort();
         // Recalculate the heap size after the sort
        reset_heap_size<<<1,1>>>();
        cudaDeviceSynchronize();
    }

    cudaFree(dev_input);
}

int main()
{
    cudaMallocManaged(&heap, MAX_HEAP_SIZE * MOL_SIZE);
    Molecule inputMolecule(to_int8(MOLECULE_STRING));
    
    int64_t soln1 = solution1(inputMolecule);
    std::cout << "Solution 1: " << soln1 << std::endl;

    solution2(inputMolecule);
    std::cout << "Solution 2: " << bestSteps << std::endl;

    cudaFree(heap);

}