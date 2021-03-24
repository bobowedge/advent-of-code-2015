import itertools

distances = open("./data/day09.input.txt").read().splitlines()
# distances = '''London to Dublin = 464
# London to Belfast = 518
# Dublin to Belfast = 141'''.splitlines()

routes = {}
cities = set()

for distance in distances:
    x = distance.split()
    city1 = x[0]
    city2 = x[2]
    d = int(x[4])
    routes[(city1, city2)] = d
    routes[(city2, city1)] = d
    cities.add(city1)
    cities.add(city2)

len_cities = len(cities)
cities = list(cities)
min_distance = None
max_distance = None
best_route = None
for route in itertools.permutations(cities):
    c = set(route)
    if len(c) != len_cities:
        continue
    d = 0
    for i in range(len_cities - 1):
        d += routes[(route[i], route[i+1])]
    if min_distance is None or d < min_distance:
        min_distance = d
        best_route = route
    if max_distance is None or d > max_distance:
        max_distance = d

print(f"Solution 1: {min_distance}")
print(f"Solution 2: {max_distance}")



