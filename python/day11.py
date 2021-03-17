alphabet = "abcdefghijklmnopqrstuvwxyz"


def is_valid(pwd):
    bad_set = set(['i','o','l'])
    pwd_set = set(list(pwd))
    if len(bad_set.intersection(pwd_set)) > 0:
        return False
    rule1 = False
    rule3 = set()
    for i, p in enumerate(list(pwd)):
        if i < len(pwd) - 1:
            if p == pwd[i + 1]:
                rule3.add(p)
            if i < len(pwd) - 2:
                ai = alphabet.index(p)
                if alphabet.index(pwd[i + 1]) == (ai + 1) and alphabet.index(pwd[i + 2]) == (ai + 2):
                    rule1 = True
        if rule1 and (len(rule3) >= 2):
            return True
    return False


def increment(pwd):
    if pwd[-1] != 'z':
        ai = alphabet.index(pwd[-1])
        return pwd[:-1] + alphabet[ai + 1]
    return increment(pwd[:-1]) + 'a'


password = "hepxcrrq"

while not is_valid(password):
    password = increment(password)

print(f"Solution 1: {password}")

password = increment(password)
while not is_valid(password):
    password = increment(password)

print(f"Solution 2: {password}")