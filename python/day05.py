import re


def rule1(s):
    vowel_count = s.count('a')
    vowel_count += s.count('e')
    vowel_count += s.count('i')
    vowel_count += s.count('o')
    vowel_count += s.count('u')
    return vowel_count > 2


def rule2(s):
    for i in range(len(s) - 1):
        if s[i + 1] == s[i]:
            return True
    return False


def rule3(s):
    if 'ab' in s or 'cd' in s or 'pq' in s or 'xy' in s:
        return False
    return True


def rule4(s):
    match = re.search(r"([a-z]{2}).*\1", s)
    return match is not None


def rule5(s):
    for i in range(len(s) - 2):
        if s[i + 2] == s[i]:
            return True
    return False


def nice1(s):
    return rule1(s) and rule2(s) and rule3(s)


def nice2(s):
    return rule4(s) and rule5(s)


# print(nice1("ugknbfddgicrmopn"),
#       nice1("aaa"),
#       nice1("jchzalrnumimnmhp"),
#       nice1("haegwjzuvuyypxyu"),
#       nice1("dvszwmarrgswjxmb"))
#
# print(nice2("qjhvhtzxzqqjkmpb"),
#       nice2("xxyxx"),
#       nice2("uurcxstgmygtbstg"),
#       nice2("ieodomkazucvgmuy"))

niceList = open("./data/day05.input.txt").read().splitlines()
nices1 = 0
nices2 = 0
for string in niceList:
    if nice1(string):
        nices1 += 1
    if nice2(string):
        nices2 += 1

print(f"Solution 1: {nices1}")
print(f"Solution 2: {nices2}")
