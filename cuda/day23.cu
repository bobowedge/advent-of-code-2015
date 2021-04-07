#include <iostream>

// Starting value for register a
__device__ __managed__ int64_t a;
// Starting value for register b
__device__ __managed__ int64_t b;

// Function pointer alias
typedef void (*fptr)(int64_t&, int64_t&);

// Cut value in half
__device__ void hlf(int64_t& x, int64_t& offset)
{
    x >>= 1;
    ++offset;
}

// Triple value
__device__ void tpl(int64_t& x, int64_t& offset)
{
    x *= 3;
    ++offset; 
}

// Increment value by 1
__device__ void inc(int64_t& x, int64_t& offset)
{
    ++x;
    ++offset;
}

// Jump offset by 19
__device__ void jmp19(int64_t&, int64_t& offset)
{
    offset += 19;
}

// Jump offset by 2
__device__ void jmp2(int64_t&, int64_t& offset)
{
    offset += 2;
}

// Jump offset by -7
__device__ void jmpneg7(int64_t&, int64_t& offset)
{
    offset -= 7;
}

// Conditional jump by 4 if value is even
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

// Conditional jump by 22 if value is 1
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

// Conditional jump by 8 if value is 1
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

// Input instructions
__device__ fptr instructions[48] = {
    jio22, inc, tpl, tpl, tpl, inc, tpl, inc, tpl, inc, inc, tpl, inc, inc, tpl,
    inc, inc, tpl, inc, inc, tpl, jmp19, tpl, tpl, tpl, tpl, inc, inc, tpl, inc,
    tpl, inc, inc, tpl, inc, inc, tpl, inc, tpl, tpl, jio8, inc, jie4, tpl, inc,
    jmp2, hlf, jmpneg7};

// Execute instructions in order
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
    // Part 1
    a = 0;
    b = 0;
    cudaDeviceSynchronize();
    execute<<<1,1>>>();
    cudaDeviceSynchronize();
    std::cout << "Solution 1: " << b << std::endl;

    // Part 2
    a = 1;
    b = 0;
    cudaDeviceSynchronize();
    execute<<<1,1>>>();
    cudaDeviceSynchronize();
    std::cout << "Solution 2: " << b << std::endl;
}