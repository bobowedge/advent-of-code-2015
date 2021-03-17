// #include <fstream>
// #include <iostream>
// #include <string>
// #include <vector>
// #include <numeric>

// #include <cryptopp/hex.h>
// #include <cryptopp/files.h>
// #define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1
// #include <cryptopp/md5.h>

// #include "aoc_utils.hh"
#include <iostream>
#include <string>
#include <vector>
#include "md5_device.hh"

const size_t THREADS = 64;
const size_t BLOCKS = 64;

__device__ __managed__ unsigned int lowest = 0xffffffff;

__device__ void itoa(int32_t N, char* Nchar, size_t enough)
{
    for (int i = enough - 1; i >=0; --i)
    {
        Nchar[i] = (N % 10) + '0';
        N /= 10;
    }
    Nchar[enough] = '\0';
}

__device__ void concat(char* str, const char* input, unsigned int input_length, 
                       const char* Nchar, unsigned int enough)
{
    int i = 0;
    for (i = 0; i < input_length; ++i)
    {
        str[i] = input[i];
    }
    for (int j = 0; j < enough; ++j, ++i)
    {
        str[i] = Nchar[j];
    }
    str[i] = '\0';
    return;
}

__device__ size_t find_enough(unsigned int N)
{
    if(N < 10)
    {
        return 1;
    }
    if(N < 100)
    {
        return 2;
    }
    if(N < 1000)
    {
        return 3;
    }
    if(N < 10000)
    {
        return 4;
    }
    if(N < 100000)
    {
        return 5;
    }
    if(N < 1000000)
    {
        return 6;
    }
    if(N < 10000000)
    {
        return 7;
    }
    if(N < 100000000)
    {
        return 8;
    }
    if(N < 1000000000)
    {
        return 9;
    }
    if(N < 10000000000)
    {
        return 10;
    }
    return 11;
}

__global__ void reset_lowest()
{
    atomicMax(&lowest, 0xffffffff);
}

__global__ void MD5_find5(char* input, unsigned int input_length)
{ 
    unsigned char hash[16];
    char Nchar[10];
    char str[16];
    size_t enough;
    unsigned int N = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int increment = blockDim.x * gridDim.x;
    while(true)
    {
        enough = find_enough(N);
        itoa(N, Nchar, enough);
        concat(str, input, input_length, Nchar, enough);
        MD5(str, hash, input_length + enough);
        if (hash[0] == 0 && hash[1] == 0 && hash[2] < 16)
        {
            atomicMin(&lowest, N);
        }
        if (N + increment >= lowest)
        {
            break;
        }
        N += increment;
    }
    return;
}

__global__ void MD5_find6(char* input, unsigned int input_length)
{ 
    unsigned char hash[16];
    char Nchar[10];
    char str[16];
    size_t enough;
    unsigned int N = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int increment = blockDim.x * gridDim.x;
    while(true)
    {
        enough = find_enough(N);
        itoa(N, Nchar, enough);
        concat(str, input, input_length, Nchar, enough);
        MD5(str, hash, input_length + enough);
        if (hash[0] == 0 && hash[1] == 0 && hash[2] == 0)
        {
            atomicMin(&lowest, N);
        }
        if (N + increment >= lowest)
        {
            break;
        }
        N += increment;
    }
    return;
}

unsigned int find_md5_zeros(const std::string& input, int64_t numZeros)
{
    char* dev_input;
    size_t dev_input_size = input.size() * sizeof(char);

    cudaMalloc((void **)&dev_input, dev_input_size);
    cudaMemcpy(dev_input, input.c_str(), dev_input_size, cudaMemcpyHostToDevice);

    reset_lowest<<<1, 1>>>();

    if (numZeros == 5)
        MD5_find5<<<BLOCKS, THREADS>>>(dev_input, input.size());
    else if (numZeros == 6)
        MD5_find6<<<BLOCKS, THREADS>>>(dev_input, input.size());

    cudaDeviceSynchronize();
    cudaFree(dev_input);
    return lowest;
}

int main()
{
    std::string test = "abcdef";
    std::cout << "Test solution: " << find_md5_zeros(test, 5) << std::endl;

    std::string input = "iwrupvqb";
    std::cout << "Solution 1: " << find_md5_zeros(input, 5) << std::endl;
    std::cout << "Solution 2: " << find_md5_zeros(input, 6) << std::endl;
}