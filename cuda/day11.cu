#include <cstdint>
#include <iostream>
#include <inttypes.h>
#include <string>

const int64_t PWDLEN = 8;
const int64_t BLOCKS = 64;
const int64_t THREADS = 64;

__device__ __managed__ int64_t nextN = 0;

__device__ bool is_valid(const char pwd[PWDLEN])
{
    bool rule2 = false;
    bool rule3 = false;
    int64_t overlap = 0;
    for (int64_t i = 0; i < PWDLEN; ++i)
    {
        if (pwd[i] == 'i' || pwd[i] == 'o' || pwd[i] == 'l')
        {
            return false;
        }
        if (i < PWDLEN - 1)
        {
            if (pwd[i] == pwd[i+1])
            {
                if (overlap != 0 && pwd[i] != overlap)
                {
                    rule3 = true;
                }
                overlap = pwd[i];
            }
            else if (i < PWDLEN - 2)
            {
                if ( (pwd[i] + 1) == pwd[i + 1] &&
                     (pwd[i+1] + 1) == pwd[i + 2])
                {
                    rule2 = true;
                }
            }
        }
    }
    return rule2 & rule3;
}

__device__ __host__ int64_t pwd_to_int(const char pwd[PWDLEN])
{
    int64_t N = 0;
    int64_t power = 1;
    for (int64_t i = PWDLEN - 1; i >= 0; --i)
    {
        N += (pwd[i] - 'a') * power;
        power *= 26;
    }
    return N;
}

__host__ int64_t pwd_to_int(const std::string pwd)
{
    return pwd_to_int(pwd.c_str());
}

__device__ void int_to_pwd(int64_t x, char pwd[PWDLEN])
{
    for (int64_t i = PWDLEN - 1; i >= 0; --i)
    {
        pwd[i] = (x % 26) + 'a';
        x /= 26;
    }
}

__host__ std::string int_to_pwd(int64_t x)
{
    std::string pwd('a', 8);
    for (int64_t i = PWDLEN - 1; i >= 0; --i)
    {
        pwd[i] = (x % 26) + 'a';
        x /= 26;
    }
    return pwd;
}

__global__ void next_pwd(char* pwd)
{
    int64_t start = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t stop = static_cast<int64_t>(__powf(26, PWDLEN));
    int64_t increment = blockDim.x * gridDim.x;
    int64_t Npwd = pwd_to_int(pwd);
    char newPwd[PWDLEN];
    for (int64_t x = start; x < stop; x += increment)
    {
        int64_t newN = Npwd + x;
        int_to_pwd(newN, newPwd);
        if (is_valid(newPwd))
        {
            atomicMin(&nextN, newN);
            break;
        }
    }
}

int main()
{
    std::string pwd = "hepxcrrq";
    nextN = 208827064576;

    char* dev_pwd;
    size_t pwdSize = PWDLEN * sizeof(char);
    cudaMalloc((void**)&dev_pwd, pwdSize);
    cudaMemcpy(dev_pwd, pwd.data(), pwdSize, cudaMemcpyHostToDevice);

    next_pwd<<<BLOCKS,THREADS>>>(dev_pwd);
    cudaDeviceSynchronize();

    std::cout << "Solution 1: " << int_to_pwd(nextN) << std::endl;

    ++nextN;
    pwd = int_to_pwd(nextN);
    nextN = 208827064576;
    cudaMemcpy(dev_pwd, pwd.data(), pwdSize, cudaMemcpyHostToDevice);

    next_pwd<<<BLOCKS,THREADS>>>(dev_pwd);
    cudaDeviceSynchronize();

    std::cout << "Solution 2: " << int_to_pwd(nextN) << std::endl;

    cudaFree(dev_pwd);
}