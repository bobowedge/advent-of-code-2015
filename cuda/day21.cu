#include <iostream>

__constant__ int WEAPON_COSTS[5] = {8, 10, 25, 40, 74};
__constant__ int WEAPON_DAMAGES[5] = {4, 5, 6, 7, 8};

__constant__ int ARMOR_COSTS[6] = {0, 13, 31, 53, 75, 102};
__constant__ int ARMOR_VALUES[6] = {0, 1, 2, 3, 4, 5};

__constant__ int RINGS_COSTS[22] = {0, 25, 50, 100, 20, 40, 80, 
    75, 125, 45, 65, 105, 150, 70, 90, 130, 120, 140, 180, 60, 100, 120};
__constant__ int RINGS_DAMAGES[22] = {0, 1, 2, 3, 0, 0, 0,
    3, 4, 1, 1, 1, 5, 2, 2, 2, 3, 3, 3, 0, 0, 0};
__constant__ int RINGS_ARMOR[22] = {0, 0, 0, 0, 1, 2, 3,
    0, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 3, 4, 5};

__constant__ int MY_HIT_POINTS = 100;
__constant__ int BOSS_HIT_POINTS = 103;
__constant__ int BOSS_DAMAGE = 9;
__constant__ int BOSS_ARMOR = 2;

const int COMBINATIONS = 5 * 6 * 22;
// const int THREADS = COMBINATIONS;
const int THREADS = 1;

__device__ __managed__ int bestCost = 500;
__device__ __managed__ int worstCost = 0;

__device__ bool win_fight(int myDamage, int myArmor)
{
    int bossHitPoints = BOSS_HIT_POINTS;
    int myHitPoints = MY_HIT_POINTS;

    int bossNetDamage = BOSS_DAMAGE - myArmor;
    int myNetDamage = myDamage - BOSS_ARMOR;
    if (bossNetDamage <= 0)
        bossNetDamage = 1;
    if (myNetDamage <= 0)
        myNetDamage = 1;

    while (bossHitPoints > 0 && myHitPoints > 0)
    {
        bossHitPoints -= myNetDamage;
        myHitPoints -= bossNetDamage;
    }
    if (bossHitPoints <= 0)
        return true;
    return false;
}

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

__global__ void find_costs()
{
    for(int N = threadIdx.x; N < COMBINATIONS; N += blockDim.x)
    {
        int myCost = 0;
        int myDamage = 0;
        int myArmor = 0;
        calculate_equipment(N, myCost, myDamage, myArmor);

        bool winner = win_fight(myDamage, myArmor);
        if (winner && myCost < bestCost)
        {
            atomicMin(&bestCost, myCost);
        }
        if (!winner && myCost > worstCost)
        {
            atomicMax(&worstCost, myCost);
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