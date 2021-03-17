sequence = "1321131112"

#sequence = "1"

sequence = [int(i) for i in sequence]
sequence.reverse()

for i in range(50):
    new_sequence = []
    x = sequence.pop()
    num = 1
    while len(sequence) > 0:
        y = sequence.pop()
        if x == y:
            num += 1
        else:
            new_sequence.append(num)
            new_sequence.append(x)
            num = 1
            x = int(y)
    new_sequence.append(num)
    new_sequence.append(x)
    sequence = list(new_sequence)
    sequence.reverse()
    if i == 39:
        print(f"Solution 1: {len(sequence)}")
    elif i == 49:
        print(f"Solution 2: {len(sequence)}")
