instructions = open("./data/day01.input.txt").read()

floor = 0
position = 1
basement = None

for instruction in list(instructions):
    if instruction == '(':
        floor += 1
    if instruction == ')':
        floor -= 1
    if floor < 0 and not basement:
        basement = int(position)
    position += 1

print(f"Solution 1: {floor}")
print(f"Solution 2: {basement}")
