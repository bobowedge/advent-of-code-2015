#include <cstdint>
#include <iostream>

const int64_t row = 2947;
const int64_t col = 3029;

int64_t entry(int64_t row, int64_t col)
{
    int64_t value = col * (col + 1) / 2;
    for(int64_t i = 0; i < row - 1; ++i)
    {
        value += col + i;
    }
    return value;
}

int main()
{
    int64_t x = entry(row, col);
    int64_t value = 20151125;
    for (int64_t i = 2; i <= x; ++i)
    {
        value *= 252533;    
        value %= 33554393;
    }
    std::cout << "Solution 1: " << value << std::endl;
}