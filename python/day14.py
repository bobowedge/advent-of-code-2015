def distance_traveled(time, v, delta, w):
    t = 0
    d = 0
    while t < time:
        if (t + delta) < time:
            t += delta + w
            d += v * delta
        else:
            d += v * (time - t)
            t = time
    return d


data = open("./data/day14.input.txt").read().splitlines()

reindeer = {}
for line in data:
    line = line.split()
    name = line[0]
    speed = int(line[3])
    duration = int(line[6])
    wait = int(line[-2])
    reindeer[name] = (speed, duration, wait)

soln1 = None
score2 = {}
for time in range(1, 2504):
    best_reindeer = []
    best_distance = None
    for name in reindeer:
        (speed, duration, wait) = reindeer[name]
        d = distance_traveled(time, speed, duration, wait)
        if best_distance is None or d > best_distance:
            best_distance = d
            best_reindeer = [name]
        elif d == best_distance:
            best_reindeer.append(name)
    for name in best_reindeer:
        score2[name] = score2.get(name, 0) + 1
    if time == 2503:
        soln1 = best_distance

soln2 = None
for name in score2:
    if soln2 is None or score2[name] > soln2:
        soln2 = score2[name]

print(f"Solution 1: {soln1}")
print(f"Solution 2: {soln2}")
