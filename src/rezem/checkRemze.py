import random

# Generate a random floating-point number between 0.0 and 1.

def func(x):
    return 1 + x - 2 ** x

def funcApprox(x):
    p = [-1.55032043e-5, -1.56598145e-4, -1.33314802e-3, -9.61658346e-3, -5.55041739e-2, -2.40226805e-1,  3.06852825e-1,  8.96880707e-9]
    val = 0
    for i in range(len(p)):
        val *= x
        val += p[i]
    return val

iter = 10
for _ in range(iter):
    x = random.random()
    print(abs(func(x) - funcApprox(x)))

