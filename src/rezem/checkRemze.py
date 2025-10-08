import random
import math
# Generate a random floating-point number between 0.0 and 1.

def func(x):
    return math.erfc(x)

import math

def funcApprox2(x):
    """
    Approximates the complementary error function erfc(x) using Abramowitz and Stegun formula 7.1.26.
    This approximation is for x >= 0.
    """
    p = 0.3275911
    a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    
    # Abramowitz and Stegun formula 7.1.26
    # t = 1 / (1 + p*x)
    t = 1.0 / (1.0 + p * x)
    
    # This loop is a Horner's method for polynomial evaluation
    s = 0.0
    for i in range(4, -1, -1):
        s = t * (s + a[i])
        
    # Complete the formula
    s *= math.exp(-x * x)
    return s

# Example usage:
# x_val = 1.0
# approx_erfc = erfc_approx(x_val)
# actual_erfc = math.erfc(x_val)
# print(f"Approximation for erfc({x_val}): {approx_erfc}")
# print(f"math.erfc({x_val}):          {actual_erfc}")

def funcApprox(x):
    p = [ 1.78383170e-02,  1.78598874e-15, -1.06823964e-01, -2.00729965e-15,
  3.74585014e-01,  4.82066302e-16, -1.12827118e+00,  1.00000000e+00]
    val = 0
    for i in range(len(p)):
        val *= x
        val += p[i]
    return val

iter = 10
for _ in range(iter):
    x = random.random() * 4
    print(abs(func(x) - funcApprox(x)), abs(func(x) - funcApprox2(x)))

