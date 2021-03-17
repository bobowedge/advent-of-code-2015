def read_ingredient(item: str):
    item = item.split()
    result = []
    for i in range(2, 10, 2):
        result.append(int(item[i][:-1]))
    result.append(int(item[10]))
    return result


data = open("./data/day15.input.txt").read().splitlines()

ingredients = []
for line in data:
    temp = read_ingredient(line)
    ingredients.append(temp)

soln1 = None
soln2 = None
for p in range(100):
    x0 = [p * i for i in ingredients[0]]
    for q in range(100-p):
        x1 = [q * i for i in ingredients[1]]
        for r in range(100-(p+q)):
            x2 = [r * i for i in ingredients[2]]
            s = 100 - (p + q + r)
            x3 = [s * i for i in ingredients[3]]
            y = [max(x0[i] + x1[i] + x2[i] + x3[i], 0) for i in range(4)]
            calories = x0[4] + x1[4] + x2[4] + x3[4]
            val = y[0] * y[1] * y[2] * y[3]
            if soln1 is None or val > soln1:
                soln1 = val
            if calories == 500 and (soln2 is None or val > soln2):
                soln2 = val

print(f"Solution 1: {soln1}")
print(f"Solution 2: {soln2}")
