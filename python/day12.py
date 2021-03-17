import json


def sum_object(obj, ignore=False):
    if type(obj) == int:
        return obj
    if type(obj) == str:
        return 0
    if type(obj) == list:
        obj_sum = 0
        for x in obj:
            obj_sum += sum_object(x, ignore)
        return obj_sum
    if type(obj) == dict:
        obj_sum = 0
        for x in obj:
            if ignore and obj[x] == "red":
                return 0
            obj_sum += sum_object(obj[x], ignore)
        return obj_sum
    raise RuntimeError("Unknown object")


json_load = json.load(open("./data/day12.input.txt"))

number_sum1 = 0
number_sum2 = 0
for y in json_load:
    number_sum1 += sum_object(y)
    number_sum2 += sum_object(y, True)
print(f"Solution 1: {number_sum1}")
print(f"Solution 2: {number_sum2}")
