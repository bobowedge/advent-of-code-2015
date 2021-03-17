dimensions = open("./data/day02.input.txt").read().splitlines()

paperNeeded = 0
ribbonNeeded = 0

for dimension in dimensions:
    dims = dimension.split("x")
    dims = [int(x) for x in dims]
    dims.sort()
    paperNeeded += 3*dims[0]*dims[1]
    paperNeeded += 2*dims[1]*dims[2]
    paperNeeded += 2*dims[0]*dims[2]

    ribbonNeeded += 2*(dims[0]+dims[1])
    ribbonNeeded += dims[0]*dims[1]*dims[2]

print(f"Solution 1: {paperNeeded}")
print(f"Solution 2: {ribbonNeeded}")
