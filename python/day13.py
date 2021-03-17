from collections import defaultdict
import itertools


def parse_data(lines):
    h = {}
    n = set()
    for line in data:
        line = line.split()
        names = [line[0], line[-1][:-1]]
        n.add(names[0])
        n.add(names[1])
        names.sort()
        names = tuple(names)
        units = int(line[3])
        if line[2] == "lose":
            units = -units
        if names in h:
            h[names] += units
        else:
            h[names] = units
    return h, n


def score_seating(s, h):
    total = 0
    for key in s:
        names = [key, s[key]]
        names.sort()
        names = tuple(names)
        total += h.get(names, 0)
    return total


def best_seating(h, n):
    best_happiness = None
    for name_list in itertools.permutations(names):
        s = {}
        for i in range(len(name_list) - 1):
            s[name_list[i]] = name_list[i + 1]
        s[name_list[-1]] = name_list[0]
        score = score_seating(s, happiness)
        if best_happiness is None or score > best_happiness:
            best_happiness = score
    return best_happiness


data = open("./data/day13.input.txt").read().splitlines()
happiness, names = parse_data(data)

soln1 = best_seating(happiness, names)
print(f"Solution 1: {soln1}")
names.add("You")
soln2 = best_seating(happiness, names)
print(f"Solution 2: {soln2}")
