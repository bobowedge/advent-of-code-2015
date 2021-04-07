#include <iostream>

// Weapon costs and damages
__constant__ int WEAPON_COSTS[5] = {8, 10, 25, 40, 74};
__constant__ int WEAPON_DAMAGES[5] = {4, 5, 6, 7, 8};

// Armor costs and values
__constant__ int ARMOR_COSTS[6] = {0, 13, 31, 53, 75, 102};
__constant__ int ARMOR_VALUES[6] = {0, 1, 2, 3, 4, 5};

// Rings costs, damage buffs, and armor buffs
__constant__ int RINGS_COSTS[22] = {0, 25, 50, 100, 20, 40, 80, 
    75, 125, 45, 65, 105, 150, 70, 90, 130, 120, 140, 180, 60, 100, 120};
__constant__ int RINGS_DAMAGES[22] = {0, 1, 2, 3, 0, 0, 0,
    3, 4, 1, 1, 1, 5, 2, 2, 2, 3, 3, 3, 0, 0, 0};
__constant__ int RINGS_ARMOR[22] = {0, 0, 0, 0, 1, 2, 3,
    0, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 3, 4, 5};

// Player starting HP    
__constant__ int PLAYER_HP = 100;
// Boss' starting HP
__constant__ int BOSS_HP = 103;
// Boss' damage
__constant__ int BOSS_DAMAGE = 9;
// Boss' armor
__constant__ int BOSS_ARMOR = 2;

// 5 weapons, 6 armors (including none), 22 ring combinations (0-2 unique rings: 1 + 6 + 15)
const int COMBINATIONS = 5 * 6 * 22;
const int THREADS = COMBINATIONS;
// const int THREADS = 1;

// Cost for Part 1 (746 = buy everything, which is not possible)
__device__ __managed__ int bestCost = 746;
// Cost for Part 2 (buy nothing)
__device__ __managed__ int worstCost = 0;

// Given an input damage and armor, decide who wins fight (true is Player)
__device__ bool win_fight(int playerDamage, int playerArmor)
{
    int bossHP = BOSS_HP;
    int playerHP = PLAYER_HP;

    int bossNetDamage = BOSS_DAMAGE - playerArmor;
    if (bossNetDamage <= 0)
    {
        bossNetDamage = 1;
    }
    int playerNetDamage = playerDamage - BOSS_ARMOR;
    if (playerNetDamage <= 0)
    {
        playerNetDamage = 1;
    }

    // Attack until at least one HP <= 0
    while (bossHP > 0 && playerHP > 0)
    {
        bossHP -= playerNetDamage;
        playerHP -= bossNetDamage;
    }

    // Player wins if bossHP <= 0
    return (bossHP <= 0);
}

// Convert number N to combination cost, damage, and armor
__device__ void calculate_equipment(int N, 
    int& cost, int& damage, int& armor)
{
    cost = 0;
    damage = 0;
    armor = 0;

    int weaponIndex = N % 5;
    damage += WEAPON_DAMAGES[weaponIndex];
    cost += WEAPON_COSTS[weaponIndex];

    int armorIndex = (N / 5) % 6;
    armor += ARMOR_VALUES[armorIndex];
    cost += ARMOR_COSTS[armorIndex];
    
    int ringIndex = (N / 30) % 22;
    damage += RINGS_DAMAGES[ringIndex];
    armor += RINGS_ARMOR[ringIndex];
    cost += RINGS_COSTS[ringIndex];
    return;
}

// Find best cost for Part 1 and worst cost for Part 2
__global__ void find_costs()
{
    int playerCost = 0;
    int playerDamage = 0;
    int playerArmor = 0;
    for(int N = threadIdx.x; N < COMBINATIONS; N += blockDim.x)
    {
        // Convert N to equipment
        calculate_equipment(N, playerCost, playerDamage, playerArmor);

        // Determine fight winner
        bool winner = win_fight(playerDamage, playerArmor);
        // Set best cost for Part 1
        if (winner && playerCost < bestCost)
        {
            atomicMin(&bestCost, playerCost);
        }
        // Set worst cost for Part 2
        if (!winner && playerCost > worstCost)
        {
            atomicMax(&worstCost, playerCost);
        }
    }
}

int main()
{
    find_costs<<<1,THREADS>>>();
    cudaDeviceSynchronize();

    std::cout << "Solution 1: " << bestCost << std::endl;
    std::cout << "Solution 2: " << worstCost << std::endl;
}