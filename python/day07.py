logics = open("./data/day07.input.txt").read().splitlines()

# logics = """123 -> x
# 456 -> y
# x AND y -> d
# x OR y -> e
# x LSHIFT 2 -> f
# y RSHIFT 2 -> g
# NOT x -> h
# NOT y -> i""".splitlines()


def compute_gates(lines):
    gates = {}
    index = 0
    while len(lines) > 0:
        index %= len(lines)
        logic = lines[index]
        (left, right) = logic.split("->")
        left = left.strip().split()
        right = right.strip()
        if right in gates:
            lines.pop(index)
        elif len(left) == 1:
            if left[0].isdigit():
                gates[right] = int(left[0])
                lines.pop(index)
            elif left[0] in gates:
                gates[right] = gates[left[0]]
                lines.pop(index)
            else:
                index += 1
        elif len(left) == 2 and left[0] == "NOT":
            if left[1].isdigit():
                gates[right] = 65535 - int(left[1])
                lines.pop(index)
            elif left[1] in gates:
                gates[right] = 65535 - gates[left[1]]
                lines.pop(index)
            else:
                index += 1
        elif len(left) == 3:
            (val1, val2) = (None, None)
            if left[0].isdigit():
                val1 = int(left[0])
            elif left[0] in gates:
                val1 = int(gates[left[0]])
            if left[2].isdigit():
                val2 = int(left[2])
            elif left[2] in gates:
                val2 = int(gates[left[2]])
            if val1 is not None and val2 is not None:
                if left[1] == "AND":
                    gates[right] = val1 & val2
                if left[1] == "OR":
                    gates[right] = val1 | val2
                if left[1] == "LSHIFT":
                    gates[right] = val1 << val2
                if left[1] == "RSHIFT":
                    gates[right] = val1 >> val2
                lines.pop(index)
            else:
                index += 1
    return gates


gates1 = compute_gates(list(logics))
print(f"Solution 1: {gates1['a']}")

value = gates1["a"]
logics2 = [f"{value} -> b"] + list(logics)
gates2 = compute_gates(list(logics2))
print(f"Solution 2: {gates2['a']}")
