/** 
 * A really janky implementation of the MD5 hash algorithm
 * It works as long as the input_length is less than 56 (bytes)
 */
#include <stdint.h>

__device__ static const unsigned int S[64] = { 
                             7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
                             5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
                             4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23, 
                             6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21};

__device__ static const int shifts[16] = { 
                             7, 12, 17, 22,  5,  9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21};

__device__ static const unsigned int K[64] = { 
                             0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
                             0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 
                             0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 
                             0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 
                             0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 
                             0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, 
                             0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 
                             0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 
                             0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 
                             0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 
                             0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 
                             0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 
                             0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 
                             0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1, 
                             0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 
                             0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

__device__ void MD5(char* input, unsigned char* hash, unsigned int input_length)
{
    unsigned int aa = 0x67452301;
    unsigned int bb = 0xefcdab89;
    unsigned int cc = 0x98badcfe;
    unsigned int dd = 0x10325476;

    unsigned char block[64];
    for (int i = 0; i < input_length; ++i)
    {
        block[i] = input[i];
    }
    block[input_length] = 0x80;
    for (int i = input_length + 1; i < 64; ++i)
    {
        block[i] = 0;
    }
    block[56] = (input_length << 3) & 0xff;
    block[57] = (input_length >> 5) & 0xff;
    block[58] = (input_length >> 13) & 0xff;
    block[59] = (input_length >> 21) & 0xff;

    unsigned int A = aa;
    unsigned int B = bb;
    unsigned int C = cc;
    unsigned int D = dd;

    unsigned int F;
    int G;
    for (int i = 0; i < 64; ++i)
    {
        if (i < 16)
        {
            F = (B & C) | (~B & D);
            G = i;
        }
        else if (i < 32)
        {
            F = (B & D) | (C & ~D);
            G = (5 * i) + 1;
        }
        else if (i < 48)
        {
            F = B ^ C ^ D;
            G = (3 * i) + 5;
        }
        else
        {
            F = C ^ (B | ~D);
            G = (7 * i);
        }
        G = (G & 0x0f) * 4;

        unsigned int hold = D;
        D = C;
        C = B;
        B = A + F + K[i] + (unsigned int)(block[G] + 
                                         (block[G + 1] << 8) + 
                                         (block[G + 2] << 16) + 
                                         (block[G + 3] << 24));
        B = (B << shifts[i & 3 | i >> 2 & ~3]) | (B >> (32 - shifts[i & 3 | i >> 2 & ~3]));
        B += C;
        A = hold;
    }
    aa += A;
    bb += B;
    cc += C;
    dd += D;

    hash[0] = aa & 0xff;
    hash[1] = (aa >> 8) & 0xff;
    hash[2] = (aa >> 16) & 0xff;
    hash[3] = (aa >> 24) & 0xff;
    hash[4] = bb & 0xff;
    hash[5] = (bb >> 8) & 0xff;
    hash[6] = (bb >> 16) & 0xff;
    hash[7] = (bb >> 24) & 0xff;
    hash[8] = cc & 0xff;
    hash[9] = (cc >> 8) & 0xff;
    hash[10] = (cc >> 16) & 0xff;
    hash[11] = (cc >> 24) & 0xff;
    hash[12] = dd & 0xff;
    hash[13] = (dd >> 8) & 0xff;
    hash[14] = (dd >> 16) & 0xff;
    hash[15] = (dd >> 24) & 0xff;
}