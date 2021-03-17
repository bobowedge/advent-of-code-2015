#include <cstdint>
#include <iostream>

const int BLOCKS = 1024;
const int THREADS = 256;

__device__ __managed__ int bestMana = INT_MAX;

class Player;

class Boss
{
public:
    int hp = 51;
    int damage = 9;
    __device__ Boss() {};

    __device__ void attack(Player& player);

    __device__ bool is_dead()
    {
        return (hp <= 0);
    }
};

class Player
{
public:
    int manaSpent = 0;
    int mana = 500;
    int hp = 50;
    int armor = 0;
    int shieldTimer = 0;
    int poisonTimer = 0;
    int rechargeTimer = 0;
    __device__ Player() {};

    __device__ bool magic_missile(Boss& boss)
    {
        boss.hp -= 4;
        mana -= 53;
        manaSpent += 53;
        return true;
    }

    __device__ bool drain(Boss& boss)
    {
        boss.hp -= 2;
        hp += 2;
        mana -= 73;
        manaSpent += 73;
        return true;
    }

    __device__ bool shield()
    {
        if (shieldTimer > 0)
            return false;
        shieldTimer = 6;
        armor = 7;
        mana -= 113;
        manaSpent += 113;
        return true;
    }

    __device__ bool poison()
    {
        if (poisonTimer > 0)
            return false;
        poisonTimer = 6;
        mana -= 173;
        manaSpent += 173;
        return true;
    }

    __device__ bool recharge()
    {
        if (rechargeTimer > 0)
            return false;
        rechargeTimer = 5;
        mana -= 229;
        manaSpent += 229;
        return true;
    }

    __device__ bool can_cast()
    {
        return (mana >= 53);
    }

    __device__ bool is_dead()
    {
        return (hp <= 0);
    }
};

__device__ void Boss::attack(Player& player)
{
    if (damage <= player.armor)
        player.hp -= 1;
    else
        player.hp -= (damage - player.armor);
}

__device__ void turn_start(Player& player, Boss& boss)
{
    if (player.shieldTimer > 0)
    {
        --player.shieldTimer;
        if (player.shieldTimer == 0)
        {
            player.armor = 0;
        }
    }
    if (player.poisonTimer > 0)
    {
        boss.hp -= 3;
        --player.poisonTimer;
    }
    if (player.rechargeTimer > 0)
    {
        player.mana += 101;
        --player.rechargeTimer;
    }
}

__device__ bool round(int& type, Player& player, Boss& boss, bool hardMode)
{
    if (hardMode)
    {
        player.hp -= 1;
        if (player.hp <= 0)
            return false;
    }

    turn_start(player, boss);

    int mod = type % 5;
    bool validRound = true;
    switch(mod)
    {
        case 0:
        {
            validRound = player.magic_missile(boss);
            break;
        }
        case 1:
        {
            validRound = player.drain(boss);
            break;
        }
        case 2:
        {
            validRound = player.shield();
            break;
        }
        case 3:
        {
            validRound = player.poison();
            break;
        }
        case 4:
        {
            validRound = player.recharge();
            break;
        }
    }
    if (!validRound)
    {
        return false;
    }

    turn_start(player, boss);
    boss.attack(player);
    type /= 5;
    return true;
}


__global__ void find_best_mana(bool hardMode)
{
    for(int64_t N = threadIdx.x + blockIdx.x * blockDim.x; N < INT_MAX; N += blockDim.x * gridDim.x)
    {
        int x = N;
        Player player;
        Boss boss;
        while(player.can_cast() && !player.is_dead() && !boss.is_dead() && player.manaSpent < bestMana)
        {
            bool validRound = round(x, player, boss, hardMode);
            if (!validRound)
                break;
        }
        if (boss.is_dead())
        {
            atomicMin(&bestMana, player.manaSpent);
        }
    }
}

int main()
{
    find_best_mana<<<BLOCKS,THREADS>>>(false);
    cudaDeviceSynchronize();
    std::cout << "Solution 1: " << bestMana << std::endl;

    bestMana = INT_MAX;
    cudaDeviceSynchronize();
    find_best_mana<<<BLOCKS,THREADS>>>(true);
    cudaDeviceSynchronize();
    std::cout << "Solution 2: " << bestMana << std::endl;
}
