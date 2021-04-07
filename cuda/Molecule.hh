#pragma once
#include <cstdint>
#include "Transforms.hh"

// Maximum size of the array in a Molecule
constexpr size_t MAX_MOL_SIZE = 285;

/**
 * A Molecule consists of 
 * - molecule - an array of integers representing elements
 * - msize - Size of molecule
 * - steps - Number of steps (from start) taken to produce this molecule (Part 2)
 */
struct Molecule
{
    // Represents string of elements
    int8_t molecule[MAX_MOL_SIZE];
    // Size of molecule
    size_t msize;
    // Number of steps taken to produce this molecule (used in Part 2)
    size_t steps;

    // Produce an invalid molecule (host and device)
    __device__ __host__ Molecule()
    {
        msize = MAX_MOL_SIZE;
        steps = INT_MAX;
    }

    // Produce a molecule from a string of elements (device only)
    __device__ __host__ Molecule(int8_t* newm, size_t newSize, size_t newSteps = 0)
    {
        for(size_t i = 0; i < newSize; ++i)
        {
            molecule[i] = newm[i];
        }
        msize = newSize;
        steps = newSteps;
    }

    // Produce a molecule from a string of elements (host only)
    __host__ Molecule(std::vector<int8_t> m, size_t s = 0) :
        Molecule(m.data(), m.size(), s)
    {
    }
    
    /** 
     * \brief Fabricate a new molecule from a given transform (by replacement)
     * 
     * \param transformIdx Index of transform in TRANSFORMS
     * \param childNum How many replacements to skip before applying transform
     * 
     * New molecule will be at least as long as the previous one
     * 
     * \return Either new molecule with replacement made or invalid molecule
     */
    __device__ Molecule fabricate(int64_t transformIdx, int64_t childNum) const
    {
        // Select transform and get its size
        int8_t* transform = TRANSFORMS[transformIdx];
        size_t tsize = TSIZES[transformIdx];

        // New molecule string
        int8_t newm[MAX_MOL_SIZE];

        int64_t newIdx = 0;
        int64_t childIdx = 0;
        for(int64_t oldIdx = 0; oldIdx < msize; ++oldIdx)
        {
            if (molecule[oldIdx] == transform[0])
            {
                // Replace with transform values
                if (childIdx == childNum)
                {
                    for (int64_t j = 1; j < tsize; ++j)
                    {
                        newm[newIdx] = transform[j];
                        ++newIdx;
                    }
                }
                // Skip this possible replacement
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
        // Check if a replacement was actually made
        if (newIdx == msize)
        {
            return Molecule();
        }
        // Create new molecule
        return Molecule(newm, newIdx, steps + 1);
    }

    /**
     * \brief Convert a molecule to a smaller one via a particular reverse transformation
     * 
     * \param transformIdx Index of transformation in TRANSFORMS
     * 
     * New molecule will be no longer than existing one
     * 
     * \return New molecule, if deconstruction worked, or invalid molecule
     */
    __device__ Molecule deconstruct(int64_t transformIdx) const
    {
        // Select transform and its size
        int8_t* transform = TRANSFORMS[transformIdx];
        size_t tsize = TSIZES[transformIdx];

        // Replacement with this transform is impossible (longer than molecule)
        if (msize + 1 < tsize)
        {
            return Molecule();
        }

        // New molecule
        int8_t m[MAX_MOL_SIZE];

        size_t thisIdx = 0;
        size_t mIdx = 0;
        while (thisIdx < msize)
        {
            bool equal = false;
            // Check if replacement is possible and not already made
            if (mIdx == thisIdx && thisIdx <= msize + 1 - tsize)
            {
                equal = true;
                for (size_t j = 1; j < tsize; ++j)
                {
                    equal &= (molecule[thisIdx + j - 1] == transform[j]);
                }
                // Replace
                if (equal)
                {
                    m[mIdx] = transform[0];
                    thisIdx += tsize - 1;
                }
            }
            // Copy the molecule if replacement not made
            if (!equal)
            {
                m[mIdx] = molecule[thisIdx];
                ++thisIdx;
            }
            ++mIdx;
        }
        // No change => invalid molecule/transform pair
        if (mIdx == thisIdx)
        {
            return Molecule();
        }
        // Create new molecule with one more step
        return Molecule(m, mIdx, steps + 1);
    }

    // Is this a valid molecule?
    __device__ bool is_valid() const
    {
        return (steps != INT_MAX && msize < MAX_MOL_SIZE);
    }
};

// Comparison used by bitonic sort - sort by size, then by number of steps
__device__ bool operator<(const Molecule& m1, const Molecule& m2)
{
    if (m1.msize != m2.msize)
    {
        return m1.msize < m2.msize;
    }
    return m1.steps < m2.steps;
}

// Comparison used by compare_unique
__device__ bool operator==(const Molecule& m1, const Molecule& m2)
{
    if (m1.msize != m2.msize)
    {
        return false;
    }
    for (int64_t i = 0; i < m1.msize; ++i)
    {
        if (m1.molecule[i] != m2.molecule[i])
        {
            return false;
        }
    }
    return true;
}


