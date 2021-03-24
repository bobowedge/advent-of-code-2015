// nvcc -g -o day04.exe day04.cu
#include <iostream>
#include <string>
#include <vector>
#include "md5_device.hh"

// Secret input key
const std::string SECRET_KEY = "iwrupvqb";

// Somewhat arbitrary choice of threads and blocks
const size_t THREADS = 64;
const size_t BLOCKS = 64;

// String size of largest integer
const size_t MAX_NCHAR_SIZE = 20;
// MAX_NCHAR_SIZE + SECRET_KEY.size()
const size_t MAX_SIZE = 28;

// Managed memory for solution
__device__ __managed__ unsigned int solution = UINT_MAX;

// Convert an integer to a C-string
__device__ void itoa(int32_t N, char* Nchar, size_t Nsize)
{
    for (int i = Nsize - 1; i >=0; --i)
    {
        Nchar[i] = (N % 10) + '0';
        N /= 10;
    }
    Nchar[Nsize] = '\0';
}

// Concatenate key and Nchar together
__device__ void concat(char* str, const char* key, unsigned int keyLength, 
                       const char* Nchar, unsigned int Nsize)
{
    int i = 0;
    for (i = 0; i < keyLength; ++i)
    {
        str[i] = key[i];
    }
    for (int j = 0; j < Nsize; ++j, ++i)
    {
        str[i] = Nchar[j];
    }
    str[i] = '\0';
    return;
}

// Reset the solution to UINT_MAX
__global__ void reset_solution()
{
    atomicMax(&solution, UINT_MAX);
}

// Calculate and find the hash with leading zeros
__global__ void MD5_find(char* key, unsigned int keyLength, int numZeros)
{ 
    // MD5 hash array
    unsigned char hash[16];
    // Array for integer converted to string
    char Nchar[MAX_NCHAR_SIZE];
    // Array for KEY + Nchar
    char str[MAX_SIZE];
    // Length of N as string
    size_t Nsize;

    // Starting integer for this thread
    unsigned int N = threadIdx.x + blockIdx.x * blockDim.x;
    // Increment to next integer for this thread
    unsigned int increment = blockDim.x * gridDim.x;
    // Loop until a solution is found
    while(true)
    {
        // Size of N as string
        Nsize = (size_t)log10((double)N) + 1;
        // Convert N to string
        itoa(N, Nchar, Nsize);
        // Concatenate SECRET_KEY and Nchar
        concat(str, key, keyLength, Nchar, Nsize);
        // Compute MD5 of concatenation
        MD5(str, hash, keyLength + Nsize);

        // Check for 5 zeros: 00 00 0X or
        // Check for 6 zeros: 00 00 00
        if (hash[0] == 0 && hash[1] == 0)
        {
            if ( (numZeros == 5 && hash[2] < 16) ||
                 (numZeros == 6 && hash[2] == 0) )
            {
                atomicMin(&solution, N);
            }
        }
        // Break if next N can't yield better solution
        if (N >= solution - increment)
        {
            break;
        }
        N += increment;
    }
    return;
}
    
unsigned int find_md5_zeros(const std::string& key, int numZeros)
{
    // Device input key
    char* dev_key;
    size_t dev_key_size = key.size() * sizeof(char);
    cudaMalloc((void **)&dev_key, dev_key_size);

    // Copy key to device
    cudaMemcpy(dev_key, key.c_str(), dev_key_size, cudaMemcpyHostToDevice);

    // Reset the solution to UINT_MAX
    reset_solution<<<1, 1>>>();

    // Find the solution
    MD5_find<<<BLOCKS, THREADS>>>(dev_key, key.size(), numZeros);

    // Sync the blocks
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(dev_key);
    return solution;
}

int main()
{
    std::cout << "Solution 1: " << find_md5_zeros(SECRET_KEY, 5) << std::endl;
    std::cout << "Solution 2: " << find_md5_zeros(SECRET_KEY, 6) << std::endl;
}