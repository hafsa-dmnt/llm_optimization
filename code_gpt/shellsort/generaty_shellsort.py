import random
import string

def generate_keys(n):
    keys = []
    for _ in range(n):
        key = ''.join(random.choices(string.printable[:94].replace(' ', ''), k=7))
        keys.append(key)
    return keys

def write_to_file(filename, n):
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        keys = generate_keys(n)
        for key in keys:
            f.write(f"{key}\n")

N = 100000000
filename = "test_input.txt"
write_to_file(filename, N)
