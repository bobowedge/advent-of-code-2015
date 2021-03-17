def move(d, x, y):
    if d == "^":
        y += 1
    elif d == "v":
        y -= 1
    elif d == "<":
        x -= 1
    else:
        x += 1
    return x, y


directions = open("./data/day03.input.txt").read()

houses1 = set()
x1 = 0
y1 = 0
houses1.add((x1, y1))

houses2 = set()
santa_x2 = 0
santa_y2 = 0
robo_x2 = 0
robo_y2 = 0
houses1.add((santa_x2, santa_y2))

for (i, direction) in enumerate(directions):
    x1, y1 = move(direction, x1, y1)
    houses1.add((x1, y1))
    if i % 2 == 0:
        santa_x2, santa_y2 = move(direction, santa_x2, santa_y2)
        houses2.add((santa_x2, santa_y2))
    else:
        robo_x2, robo_y2 = move(direction, robo_x2, robo_y2)
        houses2.add((robo_x2, robo_y2))

print(f"Solution 1: {len(houses1)}")
print(f"Solution 2: {len(houses2)}")
