import hashlib

testhash1 = "abcdef"
testhash2 = "pqrstuv"

puzzlehash = "iwrupvqb"

index = 0
solution1 = None
solution2 = None
while solution1 is None or solution2 is None:
    key = f"{puzzlehash}{index}"
    hashHex = hashlib.md5(key.encode('utf8')).hexdigest()
    if solution1 is None:
        if hashHex[:5] == "0"*5:
            solution1 = index
    else:
        if hashHex[:6] == "0"*6:
            solution2 = index
    index += 1

print(f"Solution 1: {solution1}")
print(f"Solution 2: {solution2}")

