true_sue = {"children": 3,
            "cats": 7,
            "samoyeds": 2,
            "pomeranians": 3,
            "akitas": 0,
            "vizslas": 0,
            "goldfish": 5,
            "trees": 3,
            "cars": 2,
            "perfumes": 1}

sue_lines = open("./data/day16.input.txt").read().splitlines()

sues = []
sue_num1 = None
sue_num2 = None
for sue_line in sue_lines:
    sue = {}
    colon = sue_line.find(":")
    sue_num = int(sue_line[4:colon].strip())
    sue_split = sue_line[colon + 2:].split(",")
    is_right_sue1 = True
    is_right_sue2 = True
    for data in sue_split:
        trait_val = data.split(":")
        trait = trait_val[0].strip()
        val = int(trait_val[1].strip())
        if true_sue[trait] != val:
            is_right_sue1 = False
        if trait in ["cats", "trees"]:
            if true_sue[trait] >= val:
                is_right_sue2 = False
        elif trait in ["pomeranians", "goldfish"]:
            if true_sue[trait] <= val:
                is_right_sue2 = False
        elif true_sue[trait] != val:
            is_right_sue2 = False
        sue[trait] = val
    if is_right_sue1:
        sue_num1 = sue_num
    if is_right_sue2:
        sue_num2 = sue_num

print(f"Solution 1: {sue_num1}")
print(f"Solution 2: {sue_num2}")
