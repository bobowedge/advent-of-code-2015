#include <cstdint>

const size_t MAX_MOL_SIZE = 285;

struct Molecule
{
    __device__ __host__ Molecule()
    {
        msize = MAX_MOL_SIZE;
        steps = INT_MAX;
    }

    __device__ __host__ Molecule(int64_t* m, size_t m_size, size_t s = 0)
    {
        for(size_t i = 0; i < m_size; ++i)
        {
            molecule[i] = m[i];
        }
        msize = m_size;
        steps = s;
    }

    __host__ Molecule(std::vector<int64_t> m, size_t s = 0) :
        Molecule(m.data(), m.size(), s)
    {
    }
    
    __device__ Molecule fabricate(int64_t transformIdx, int64_t childNum) const
    {
        int64_t* transform = TRANSFORMS[transformIdx];
        size_t t_size = TSIZES[transformIdx];

        int64_t newm[MAX_MOL_SIZE];

        int64_t newIdx = 0;
        int64_t childIdx = 0;
        for(int64_t oldIdx = 0; oldIdx < msize; ++oldIdx)
        {
            if (molecule[oldIdx] == transform[0])
            {
                if (childIdx == childNum)
                {
                    for (int64_t j = 1; j < t_size; ++j)
                    {
                        newm[newIdx] = transform[j];
                        ++newIdx;
                    }
                }
                else
                {
                    newm[newIdx] = molecule[oldIdx];
                    ++newIdx;
                }
                ++childIdx;
            }
            else
            {
                newm[newIdx] = molecule[oldIdx];
                ++newIdx;
            }
        }
        if (newIdx == msize)
        {
            return Molecule();
        }
        return Molecule(newm, newIdx, steps + 1);
    }

    __device__ Molecule deconstruct(int64_t transformIdx) const
    {
        int64_t* transform = TRANSFORMS[transformIdx];
        size_t t_size = TSIZES[transformIdx];

        if (msize + 1 < t_size)
            return Molecule();

        int64_t m[MAX_MOL_SIZE];

        size_t thisIdx = 0;
        size_t mIdx = 0;
        while (thisIdx < msize)
        {
            bool equal = false;
            if (mIdx == thisIdx && thisIdx <= msize + 1 - t_size)
            {
                equal = true;
                for (size_t j = 1; j < t_size; ++j)
                {
                    equal &= (molecule[thisIdx + j - 1] == transform[j]);
                }
            }
            if (equal)
            {
                m[mIdx] = transform[0];
                thisIdx += t_size - 1;
            }
            else
            {
                m[mIdx] = molecule[thisIdx];
                ++thisIdx;
            }
            ++mIdx;
        }
        if (mIdx == thisIdx)
        {
            return Molecule();
        }
        return Molecule(m, mIdx, steps + 1);
    }

    int64_t molecule[MAX_MOL_SIZE];
    size_t msize;
    size_t steps;
};

__device__ bool operator<(const Molecule& m1, const Molecule& m2)
{
    if (m1.msize != m2.msize)
        return m1.msize < m2.msize;
    return m1.steps < m2.steps;
}

__device__ bool operator==(const Molecule& m1, const Molecule& m2)
{
    if (m1.msize != m2.msize)
        return false;
    for (int64_t i = 0; i < m1.msize; ++i)
    {
        if (m1.molecule[i] != m2.molecule[i])
        {
            return false;
        }
    }
    return true;
}