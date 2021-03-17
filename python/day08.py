import re

strings = open("./data/day08.input.txt").read()
strings = strings.replace(" ", "").splitlines()
code_length = 0
char_length = 0
repr_length = 0
for string in strings:
    code_length += len(string)
    s = string[1:-1]
    s = s.replace('\\\\', '\\')
    s = re.sub(r'\\x[0-9a-f]{2}', 'b', s)
    s = s.replace('\\"', r'"')
    char_length += len(s)
    repr_length += len(string) + 2
    repr_length += string.count('"')
    repr_length += string.count('\\')

print("Solution 1:", code_length - char_length)
print("Solution 2:", repr_length - code_length)
