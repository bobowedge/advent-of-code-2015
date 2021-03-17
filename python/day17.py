def count1(s, t=150, c=0, index=0):
    if c == t:
        return 1
    if c > t or index == len(s):
        return 0
    index += 1
    return count1(s, t, c, index) + count1(s, t, c + s[index - 1], index)


def count2(s, t=150, d={}, c=0, number=0, index=0):
    if c == t:
        d[number] = d.get(number, 0) + 1
        return d
    if c > t or index == len(s):
        return d
    index += 1
    d = count2(s, t, d, c, number, index)
    d = count2(s, t, d, c + s[index-1], number + 1, index)
    return d


def best2(s, total=150):
    d = count2(s, total)
    v = min(d.keys())
    return d[v]


sizes = [43, 3, 4, 10, 21, 44, 4, 6, 47,
         41, 34, 17, 17, 44, 36, 31, 46,
         9, 27, 38]

# sizes = [20, 15, 10, 5, 5]

sizes.sort()

soln1 = count1(sizes)
print(f"Solution 1: {soln1}")

soln2 = best2(sizes)
print(f"Solution 2: {soln2}")
