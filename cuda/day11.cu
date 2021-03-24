#include <cstdint>
#include <iostream>
#include <inttypes.h>
#include <string>

// Arbitrary choice of blocks and threads
const int64_t BLOCKS = 64;
const int64_t THREADS = 64;

// Fixed password length
const int64_t PWDLEN = 8;

// Solution integer
__device__ __managed__ int64_t soln = 0;

// Check if a given password meets the rules
__device__ bool is_valid(const char pwd[PWDLEN])
{
    bool rule1 = false;
    bool rule3 = false;
    int64_t rule3Value = 0;
    for (int64_t i = 0; i < PWDLEN; ++i)
    {
        // Rule 2
        if (pwd[i] == 'i' || pwd[i] == 'o' || pwd[i] == 'l')
        {
            return false;
        }
        if (i < PWDLEN - 1)
        {
            // Rule 3
            if (pwd[i] == pwd[i+1])
            {
                if (rule3Value != 0 && pwd[i] != rule3Value)
                {
                    rule3 = true;
                }
                rule3Value = pwd[i];
            }
            // Rule 1
            else if (i < PWDLEN - 2)
            {
                if ( (pwd[i] + 1) == pwd[i + 1] &&
                     (pwd[i+1] + 1) == pwd[i + 2])
                {
                    rule1 = true;
                }
            }
        }
    }
    return rule1 & rule3;
}

// Convert an integer to a password on the device
__device__ void int_to_pwd(int64_t x, char pwd[PWDLEN])
{
    for (int64_t i = PWDLEN - 1; i >= 0; --i)
    {
        pwd[i] = (x % 26) + 'a';
        x /= 26;
    }
}

// Find the next valid password
__global__ void next_valid_pwd(int64_t intPwd)
{
    // Buffer for storing password string
    char pwd[PWDLEN];
    for (int64_t N = intPwd + threadIdx.x + blockIdx.x * blockDim.x; N < soln; N +=  blockDim.x * gridDim.x)
    {
        // Convert password integer to string
        int_to_pwd(N, pwd);
        // Check validity of password
        if (is_valid(pwd))
        {
            atomicMin(&soln, N);
            break;
        }
    }
}

// Convert a password to an integer on the host
__host__ int64_t pwd_to_int(const std::string pwd)
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

// Convert an integer to a password on the host
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

int main()
{
    // Input value
    std::string pwd = "hepxcrrq";
    // Convert to integer
    int64_t intPwd = pwd_to_int(pwd);

    // Max possible solution (corresponds to "zzzzzzzz")
    soln = static_cast<int64_t>(powf(26, PWDLEN));

    next_valid_pwd<<<BLOCKS,THREADS>>>(intPwd);
    cudaDeviceSynchronize();

    std::cout << "Solution 1: " << int_to_pwd(soln) << std::endl;

    // Increment above previous password
    intPwd = soln + 1;
    // Max possible new solution
    soln = static_cast<int64_t>(powf(26, PWDLEN));

    next_valid_pwd<<<BLOCKS,THREADS>>>(intPwd);
    cudaDeviceSynchronize();

    std::cout << "Solution 2: " << int_to_pwd(soln) << std::endl;
}