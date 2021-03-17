#include <cstdint>
#include <iostream>
#include <inttypes.h>

const int BLOCKS = 1024;
const int THREADS = 256;

const int LENGTH = 28;

__constant__ int WEIGHTS[LENGTH] = {
    1, 3, 5, 11, 13, 17, 19, 23, 29, 31, 37, 
    41, 43, 47, 53, 59, 67, 71, 73, 79, 83, 
    89, 97, 101, 103, 107, 109, 113};

// 2**28 "possible" first groups
const int64_t POSSIBLES = 1 << LENGTH;
const int TOTAL_WEIGHT = 1524;
const int GROUP_WEIGHT1 = TOTAL_WEIGHT / 3;
const int GROUP_WEIGHT2 = TOTAL_WEIGHT / 4;

__device__ __managed__ int64_t* heap;
__device__ __managed__ size_t heapSize = 0;
__device__ __managed__ int64_t* heap2;
__device__ __managed__ size_t heap2Size = 0;
__device__ __managed__ size_t heapIdx = 0;

__device__ __managed__ uint32_t bestSize = INT_MAX;
__device__ __managed__ uint64_t bestQE = 0xFFFFFFFFFFFFFFFF;

__device__ int first_weights(int64_t N, int weights[LENGTH])
{
    int length = 0;
    for(size_t i = 0; i < LENGTH; ++i)
    {
        weights[i] = 0;
        if (N % 2)
        {
            weights[i] = WEIGHTS[i];
            ++length;
        }
        N /= 2;
    }
    return length;
}

__device__ int second_weights(int64_t M, int weights1[LENGTH],
int weights2[LENGTH])
{
    int length = 0;
    for(size_t i = 0; i < LENGTH; ++i)
    {
        weights2[i] = 0;
        if (weights1[i] == 0)
        {
            if (M % 2)
            {
                weights2[i] = WEIGHTS[i];
                ++length;
            }
            M /= 2;
        }
    }
    return length;
}

__device__ void stats(int weights[LENGTH], int& weight, int64_t& qe)
{
    weight = 0;
    qe = 1;
    for(size_t i = 0; i < LENGTH; ++i)
    {
        if (weights[i] > 0)
        {
            weight += weights[i];
            qe *= weights[i];
        }
    }
    return;
}

__global__ void soln1_step1()
{
    int weights1[LENGTH];
    int numWeights1 = 0;
    int weight1 = 0;
    int64_t qe1 = 0;
    for(int64_t N = threadIdx.x + blockDim.x * blockIdx.x; N < POSSIBLES; N += blockDim.x * gridDim.x)
    {   
        numWeights1 = first_weights(N, weights1);
        if (numWeights1 > 9 || numWeights1 < 5)
        {
            continue;
        }

        stats(weights1, weight1, qe1);
        if (weight1 != GROUP_WEIGHT1)
        {
            continue;
        }
        atomicAdd(&heapSize, (size_t)1);
    }
}

__global__ void soln1_step2()
{
    int weights1[LENGTH];
    int numWeights1 = 0;
    int weight1 = 0;
    int64_t qe1 = 0;
    size_t idx = 0;
    for(int64_t N = threadIdx.x + blockDim.x * blockIdx.x; N < POSSIBLES; N += blockDim.x * gridDim.x)
    {   
        numWeights1 = first_weights(N, weights1);
        if (numWeights1 > 9 || numWeights1 < 5)
        {
            continue;
        }

        stats(weights1, weight1, qe1);
        if (weight1 != GROUP_WEIGHT1)
        {
            continue;
        }
        idx = atomicAdd(&heapIdx, (size_t)1);
        heap[idx] = N;
    }
}

__global__ void soln1_step3()
{
    int weights1[LENGTH];
    int weights2[LENGTH];
    int numWeights1 = 0;
    int weight1 = 0;
    int64_t qe1 = 0;
    int numWeights2 = 0;
    int weight2 = 0;
    int64_t qe2 = 0;
    int numWeights3 = 0;
    for(int64_t N = threadIdx.x + blockDim.x * blockIdx.x; N < POSSIBLES; N += blockDim.x * gridDim.x)
    {   
        numWeights2 = first_weights(N, weights2);
        if (numWeights2 < 5)
        {
            continue;
        }
        stats(weights2, weight2, qe2);
        if (weight2 != GROUP_WEIGHT1)
        {
            continue;
        }
        for (int64_t idx = 0; idx < heapSize; ++idx)
        {
            int64_t M = heap[idx];
            int64_t Z = M & N;
            if (Z > 0)
            {
                continue;
            }

            numWeights1 = first_weights(M, weights1);
            if (numWeights1 > bestSize)
            {
                break;
            }
            numWeights3 = LENGTH - numWeights2 - numWeights1;
            if (numWeights2 < numWeights1 || numWeights3 <= numWeights1)
            {
                continue;
            }

            stats(weights1, weight1, qe1);
            if (numWeights1 == bestSize && qe1 >= bestQE)
            {
                break;
            }
            if ((numWeights1 < bestSize) ||
                ((numWeights1 == bestSize) && qe1 < bestQE))
            {
                atomicMin(&bestSize, numWeights1);
                atomicMin(&bestQE, qe1);
                break;
            }
        }
    }
}

__global__ void soln2_step1()
{
    int weights1[LENGTH];
    int numWeights1 = 0;
    int weight1 = 0;
    int64_t qe1 = 0;
    for(int64_t N = threadIdx.x + blockDim.x * blockIdx.x; N < POSSIBLES; N += blockDim.x * gridDim.x)
    {   
        numWeights1 = first_weights(N, weights1);
        if (numWeights1 > 7 || numWeights1 < 4)
        {
            continue;
        }

        stats(weights1, weight1, qe1);
        if (weight1 != GROUP_WEIGHT2)
        {
            continue;
        }
        atomicAdd(&heapSize, (size_t)1);
    }
}

__global__ void soln2_step2()
{
    int weights1[LENGTH];
    int numWeights1 = 0;
    int weight1 = 0;
    int64_t qe1 = 0;
    size_t idx = 0;
    for(int64_t N = threadIdx.x + blockDim.x * blockIdx.x; N < POSSIBLES; N += blockDim.x * gridDim.x)
    {   
        numWeights1 = first_weights(N, weights1);
        if (numWeights1 > 7 || numWeights1 < 4)
        {
            continue;
        }

        stats(weights1, weight1, qe1);
        if (weight1 != GROUP_WEIGHT2)
        {
            continue;
        }
        idx = atomicAdd(&heapIdx, (size_t)1);
        heap[idx] = N;
    }
}

__global__ void soln2_step3()
{
    int weights1[LENGTH];
    int weights2[LENGTH];
    int numWeights1 = 0;
    int numWeights2 = 0;
    int weight2 = 0;
    int64_t qe2 = 0;
    for(int64_t N = threadIdx.x + blockDim.x * blockIdx.x; N < POSSIBLES; N += blockDim.x * gridDim.x)
    {   
        numWeights2 = first_weights(N, weights2);
        if (numWeights2 < 4)
        {
            continue;
        }
        stats(weights2, weight2, qe2);
        if (weight2 != GROUP_WEIGHT2)
        {
            continue;
        }
        bool validN = false;
        for (int64_t idx = 0; idx < heapSize; ++idx)
        {
            int64_t M = heap[idx];
            int64_t Z = M & N;
            if (Z > 0)
            {
                continue;
            }
            numWeights1 = first_weights(M, weights1);
            if (numWeights2 < numWeights1)
            {
                continue;
            }
            validN = true;
            break;
        }
        if (validN)
        {
            atomicAdd(&heap2Size, 1);
        }
    }
}

__global__ void soln2_step4()
{
    int weights1[LENGTH];
    int weights2[LENGTH];
    int numWeights1 = 0;
    int numWeights2 = 0;
    int weight2 = 0;
    int64_t qe2 = 0;
    size_t tmpIdx = 0;
    for(int64_t N2 = threadIdx.x + blockDim.x * blockIdx.x; N2 < POSSIBLES; N2 += blockDim.x * gridDim.x)
    {   
        numWeights2 = first_weights(N2, weights2);
        if (numWeights2 < 4)
        {
            continue;
        }
        stats(weights2, weight2, qe2);
        if (weight2 != GROUP_WEIGHT2)
        {
            continue;
        }
        bool validN = false;
        for (int64_t idx1 = 0; idx1 < heapSize; ++idx1)
        {
            int64_t N1 = heap[idx1];
            int64_t Z12 = N1 & N2;
            if (Z12 > 0)
            {
                continue;
            }
            numWeights1 = first_weights(N1, weights1);
            if (numWeights2 < numWeights1)
            {
                continue;
            }
            validN = true;
            break;
        }
        if (validN)
        {
            tmpIdx = atomicAdd(&heapIdx, 1);
            heap2[tmpIdx] = N2;
        }
    }
}

__global__ void soln2_step5()
{
    int weights1[LENGTH];
    int weights2[LENGTH];
    int weights3[LENGTH];
    int numWeights1 = 0;
    int weight1 = 0;
    int64_t qe1 = 0;
    int numWeights2 = 0;
    int numWeights3 = 0;
    int numWeights23 = 0;
    int numWeights4 = 0;

    int64_t N1 = 0;
    int64_t N2 = 0;
    int64_t N3 = 0;

    int64_t Z23 = 0;

    for(int64_t idx3 = blockIdx.x; idx3 < heap2Size; idx3 += gridDim.x)
    {   
        N3 = heap2[idx3];
        numWeights3 = first_weights(N3, weights3);
        for(int64_t idx2 = threadIdx.x; idx2 < heap2Size; idx2 += blockDim.x)
        {
            N2 = heap2[idx2];
            Z23 = N3 & N2;
            if (Z23 > 0)
            {
                continue;
            }
            numWeights2 = first_weights(N2, weights2);
            if (numWeights2 > numWeights3)
            {
                continue;
            }
            numWeights23 = numWeights3 + numWeights3;
            for (int64_t idx1 = 0; idx1 < heapSize; ++idx1)
            {
                N1 = heap[idx1];
                numWeights1 = first_weights(N1, weights1);
                if (numWeights1 > bestSize)
                {
                    break;
                }
                if (numWeights3 < numWeights1)
                {
                    continue;
                }
                stats(weights1, weight1, qe1);
                if (numWeights1 == bestSize && qe1 >= bestQE)
                {
                    break;
                }
                numWeights4 = LENGTH - numWeights1 - numWeights23;
                if (numWeights4 < numWeights1)
                {
                    continue;
                }
                if ((numWeights1 < bestSize) ||
                    ((numWeights1 == bestSize) && qe1 < bestQE))
                {
                    atomicMin(&bestSize, numWeights1);
                    atomicMin(&bestQE, qe1);
                    break;
                }
            }
        }
    }
}

__device__ bool compare(int64_t N1, int64_t N2)
{
    if (N1 == N2 || N1 == INT_MAX)
        return false;
    if (N2 == INT_MAX)
        return true;

    int weights1[LENGTH];
    int weights2[LENGTH]; 
    first_weights(N1, weights1);
    first_weights(N2, weights2);

    int weight1 = 0;
    int weight2 = 0;
    int64_t qe1 = 0;
    int64_t qe2 = 0;
    stats(weights1, weight1, qe1);
    stats(weights2, weight2, qe2);
    return ((weight1 < weight2) ||
            (weight1 == weight2) && qe1 < qe2);
}

__global__ void bitonic_sort_step(int64_t j, int64_t k)
{
    int64_t i = threadIdx.x + blockDim.x * blockIdx.x;
    int64_t ij = i ^ j;
    if (ij > i)
    {
        int64_t ik = i&k;
        if ( (ik == 0 && compare(heap[ij], heap[i])) ||
             (ik != 0 && compare(heap[i], heap[ij])) )
        {
            int64_t temp = heap[i];
            heap[i] = heap[ij];
            heap[ij] = temp;
        }
    }
}

__host__ void bitonic_sort(int64_t nextPow2)
{
    for (int64_t x = heapSize; x < nextPow2; ++x)
    {
        heap[x] = INT_MAX;
    }
    int64_t threads = 256;
    int64_t blocks = nextPow2 / threads;

    for (int64_t k = 2; k <= nextPow2; k <<= 1)
    {
        for(int64_t j = k >> 1; j > 0; j = j >> 1)
        {
            bitonic_sort_step<<<blocks,threads>>>(j, k);
        }
    }
    cudaDeviceSynchronize();
}


void solution1()
{
    heapSize = 0;
    heap2Size = 0;
    heapIdx = 0;
    bestSize = INT_MAX;
    bestQE = 0xFFFFFFFFFFFFFFFF;

    soln1_step1<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();

    int64_t nextPow2 = pow(2, ceil(log(heapSize)/log(2)));
    cudaMallocManaged(&heap, nextPow2 * sizeof(int64_t));

    soln1_step2<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();

    bitonic_sort(nextPow2);
    cudaDeviceSynchronize();

    soln1_step3<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();

    std::cout << "Solution 1: " << bestQE << std::endl;
}

void solution2()
{
    heapSize = 0;
    heap2Size = 0;
    heapIdx = 0;
    bestSize = INT_MAX;
    bestQE = 0xFFFFFFFFFFFFFFFF;

    soln2_step1<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();

    int64_t nextPow2 = pow(2, ceil(log(heapSize)/log(2)));
    cudaMallocManaged(&heap, nextPow2 * sizeof(int64_t));

    soln2_step2<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();
    
    soln2_step3<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();

    heapIdx = 0;
    cudaMallocManaged(&heap2, heap2Size * sizeof(int64_t));

    soln2_step4<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();

    bitonic_sort(nextPow2);

    soln2_step5<<<BLOCKS,THREADS>>>();
    cudaDeviceSynchronize();

    std::cout << "Solution 2: " << bestQE << std::endl;
}

int main()
{
    solution1();
    solution2();
    cudaFree(heap);
    cudaFree(heap2);
}