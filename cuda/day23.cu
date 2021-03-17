#include <iostream>

__device__ __managed__ int64_t a;
__device__ __managed__ int64_t b;

typedef void (*fptr)(int64_t&, int64_t&);

__device__ void hlf(int64_t& x, int64_t& offset)
{
    x >>= 1;
    ++offset;
}

__device__ void tpl(int64_t& x, int64_t& offset)
{
    x *= 3;
    ++offset; 
}

__device__ void inc(int64_t& x, int64_t& offset)
{
    ++x;
    ++offset;
}

__device__ void jmp19(int64_t&, int64_t& offset)
{
    offset += 19;
}

__device__ void jmp2(int64_t&, int64_t& offset)
{
    offset += 2;
}

__device__ void jmpneg7(int64_t&, int64_t& offset)
{
    offset -= 7;
}

__device__ void jie4(int64_t& x, int64_t& offset)
{
    if (x % 2 == 0)
    {
        offset += 4;
    }
    else
    {
        ++offset;
    }
}

__device__ void jio22(int64_t& x, int64_t& offset)
{
    if (x == 1)
    {
        offset += 22;
    }
    else
    {
        ++offset;
    }
}

__device__ void jio8(int64_t& x, int64_t& offset)
{
    if (x == 1)
    {
        offset += 8;
    }
    else
    {
        ++offset;
    }
}

__device__ fptr instructions[48] = {
    jio22, inc, tpl, tpl, tpl, inc, tpl, inc, tpl, inc, inc, tpl, inc, inc, tpl,
    inc, inc, tpl, inc, inc, tpl, jmp19, tpl, tpl, tpl, tpl, inc, inc, tpl, inc,
    tpl, inc, inc, tpl, inc, inc, tpl, inc, tpl, tpl, jio8, inc, jie4, tpl, inc,
    jmp2, hlf, jmpneg7};


__global__ void execute()
{
    int64_t offset = 0;
    while(offset >= 0 && offset < 48)
    {
        auto y = instructions[offset];
        if (offset != 41)
        {
            y(a, offset);
        }
        else
        {
            y(b, offset);
        }
    }
}

int main()
{
    a = 0;
    b = 0;
    cudaDeviceSynchronize();
    execute<<<1,1>>>();
    cudaDeviceSynchronize();
    std::cout << "Solution 1: " << b << std::endl;

    a = 1;
    b = 0;
    cudaDeviceSynchronize();
    execute<<<1,1>>>();
    cudaDeviceSynchronize();
    std::cout << "Solution 2: " << b << std::endl;
}