import re


def toggle1(z):
    return z ^ 1


def toggle2(z):
    return z + 2


def turn_off1(z):
    return 0


def turn_off2(z):
    return max(z-1, 0)


def turn_on1(z):
    return 1


def turn_on2(z):
    return z + 1


def number_on(lights):
    count = 0
    for i in range(1000):
        for j in range(1000):
            if lights[i][j] == 1:
                count += 1
    return count


def brightness(lights):
    b = 0
    for row in lights:
        for col in row:
            b += col
    return b


lights1 = [[0 for x in range(1000)] for y in range(1000)]
lights2 = [[0 for x in range(1000)] for y in range(1000)]
with open("./data/day06.input.txt") as file:
    for line in file:
        match = re.search(r"turn on (\d+),(\d+) through (\d+),(\d+)", line)
        if match:
            f = turn_on1
            g = turn_on2
        else:
            match = re.search(r"toggle (\d+),(\d+) through (\d+),(\d+)", line)
            if match:
                f = toggle1
                g = toggle2
            else:
                match = re.search(r"turn off (\d+),(\d+) through (\d+),(\d+)", line)
                f = turn_off1
                g = turn_off2
        x0 = int(match.group(1))
        y0 = int(match.group(2))
        x1 = int(match.group(3))
        y1 = int(match.group(4))
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                lights1[x][y] = f(lights1[x][y])
                lights2[x][y] = g(lights2[x][y])

print(f"Solution 1: {number_on(lights1)}")
print(f"Solution 2: {brightness(lights2)}")
