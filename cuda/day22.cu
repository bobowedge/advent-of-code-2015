#include <cstdint>
#include <iostream>

// Somewhat arbitrary choice of threads and blocks
const int BLOCKS = 1024;
const int THREADS = 256;

// Lowest amount of mana spent
__device__ __managed__ int bestMana = INT_MAX;

// Forward declare Player class
class Player;

// Boss class
class Boss
{
public:
    // HP (51 is given input)
    int hp = 51;
    // Damage (9 is given input)
    int damage = 9;

    // Constructor
    __device__ Boss() {};

    // Boss attack player for damage
    __device__ void attack(Player& player);

    // Check if the boss is dead
    __device__ bool is_dead()
    {
        return (hp <= 0);
    }
};

// Player class
class Player
{
public:
    // Total mana spent so far
    int manaSpent = 0;
    // Mana pool (500 is given input)
    int mana = 500;
    // HP (50 is given input)
    int hp = 50;
    // Armor
    int armor = 0;
    // Timer for shield
    int shieldTimer = 0;
    // Timer for poison
    int poisonTimer = 0;
    // Timer for recharge
    int rechargeTimer = 0;

    // Constructor
    __device__ Player() {};

    // Magic missile spell
    __device__ bool magic_missile(Boss& boss)
    {
        boss.hp -= 4;
        mana -= 53;
        manaSpent += 53;
        return true;
    }

    // Drain spell
    __device__ bool drain(Boss& boss)
    {
        boss.hp -= 2;
        hp += 2;
        mana -= 73;
        manaSpent += 73;
        return true;
    }

    // Shield spell
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

    // Poison spell
    __device__ bool poison()
    {
        if (poisonTimer > 0)
            return false;
        poisonTimer = 6;
        mana -= 173;
        manaSpent += 173;
        return true;
    }

    // Recharge spell
    __device__ bool recharge()
    {
        if (rechargeTimer > 0)
            return false;
        rechargeTimer = 5;
        mana -= 229;
        manaSpent += 229;
        return true;
    }

    // Check if there's enough mana to cast any spell
    __device__ bool can_cast()
    {
        return (mana >= 53);
    }

    // Check if player is dead
    __device__ bool is_dead()
    {
        return (hp <= 0);
    }
};

// Boss attack player
__device__ void Boss::attack(Player& player)
{
    if (damage <= player.armor)
    {
        player.hp -= 1;
    }
    else
    {
        player.hp -= (damage - player.armor);
    }
}

// Turn start: handle spell timers, poison, etc.
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

// Execute player turn (timers, poison, player spell)
// Returns true if cast was succesful, false otherwise
__device__ bool player_turn(int spell, Player& player, Boss& boss, bool hardMode)
{
    // On hard mode, player loses 1 HP every player turn
    if (hardMode)
    {
        player.hp -= 1;
        if (player.hp <= 0)
        {
            return false;
        }
    }
    // Run turn start mechanics
    turn_start(player, boss);
    // Cast spell
    switch(spell)
    {
        case 0:
        {
            return player.magic_missile(boss);
        }
        case 1:
        {
            return player.drain(boss);
        }
        case 2:
        {
            return player.shield();
        }
        case 3:
        {
            return player.poison();
        }
        case 4:
        {
            return player.recharge();
        }
    }
    // Should never reach here
    return false;
}

// Execute boss turn (timers, poison, boss attack)
__device__ void boss_turn(Player& player, Boss& boss)
{
    // Turn start mechanics
    turn_start(player, boss);
    // Boss attack
    boss.attack(player);
}

// Execute one round, given spell list (return true if player could cast spell, false if not)
__device__ bool round(int64_t& spellList, Player& player, Boss& boss, bool hardMode)
{
    // Get next spell from list
    int spell = spellList % 5;
    // Take player turn
    bool valid = player_turn(spell, player, boss, hardMode);
    // Invalid round (not enough mana to cast the next spell)
    if (!valid)
    {
        return false;
    }
    // Take boss turn
    boss_turn(player, boss);
    // Discard last spell
    spellList /= 5;
    // Valid round
    return true;
}

// Find the smallest mana to use to beat boss
__global__ void find_best_mana(bool hardMode)
{
    int64_t spellList = 0;
    Player player;
    Boss boss;
    for(int64_t N = threadIdx.x + blockIdx.x * blockDim.x; N < INT_MAX; N += blockDim.x * gridDim.x)
    {
        spellList = N;
        player = Player();
        boss = Boss();
        // Continue until boss is dead, player is dead, or too much mana is spent
        while(player.can_cast() && !player.is_dead() && !boss.is_dead() && player.manaSpent < bestMana)
        {
            // Run a single round (player turn + boss turn)
            if (!round(spellList, player, boss, hardMode))
            {
                break;
            }
        }
        // If boss is dead, check against best mana
        if (boss.is_dead())
        {
            atomicMin(&bestMana, player.manaSpent);
        }
    }
}

int main()
{
    // Part 1
    find_best_mana<<<BLOCKS,THREADS>>>(false);
    cudaDeviceSynchronize();
    std::cout << "Solution 1: " << bestMana << std::endl;

    // Reset bestMana
    bestMana = INT_MAX;
    cudaDeviceSynchronize();

    // Part 2
    find_best_mana<<<BLOCKS,THREADS>>>(true);
    cudaDeviceSynchronize();
    std::cout << "Solution 2: " << bestMana << std::endl;
}
