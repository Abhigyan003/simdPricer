# AI generated, reference from https://dkenefake.github.io/blog/Optimal_Poly

import numpy
import math

def chev_points(n):
    """
    Computes the Chebyshev nodes (extrema of the first-kind Chebyshev polynomial).
    
    Args:
        n (int): The number of points to generate. This corresponds to the
                 polynomial degree + 2 in the Remez algorithm.

    Returns:
        numpy.ndarray: An array of n Chebyshev nodes in the interval [-1, 1].
    """
    # Create an array for k from 0 to n-1
    k = numpy.arange(n)
    
    # Calculate the points using the standard formula
    # These points are the roots of T_n'(x), or the extrema of T_n(x)
    points = numpy.cos(k * numpy.pi / (n - 1))
    
    # The points are generated in descending order [1, ..., -1], which is fine.
    # If ascending order is needed, they can be flipped with numpy.flip().
    return points

def bisection_search(f, low, high, tol=1e-12):
    """
    Finds a root of function f within the interval [low, high]
    using the bisection method. Assumes a root is bracketed.
    """
    if f(low) * f(high) > 0:
        # Fallback if root is not bracketed, though this shouldn't happen
        # in the Remez algorithm if points are chosen correctly.
        return (low + high) / 2.0

    mid = 0.0
    while (high - low) / 2.0 > tol:
        mid = (low + high) / 2.0
        if f(mid) == 0:
            return mid
        elif f(low) * f(mid) < 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0

def concave_max(f, low, high):
    """
    Finds the local extremum (max or min) of function f within the
    interval [low, high] by finding the root of its derivative.
    """
    # Create an approximate derivative expression using central difference
    h = 1e-6
    df = lambda x: (f(x + h) - f(x - h)) / (2.0 * h)
    
    # Find the root of the derivative, which corresponds to an extremum
    return bisection_search(df, low, high)

def func(x):
    """The target function for approximation."""
    return math.erfc(x)

def remez(func, n_degree):
    """
    Calculates the best polynomial approximation of a given degree
    for the specified function using the Remez algorithm.
    """
    # Initialize the node points using Chebyshev nodes
    # We need n+2 points for a polynomial of degree n
    x_points = chev_points(n_degree + 2)
    max_iter = 100
    
    A = numpy.zeros((n_degree + 2, n_degree + 2))
    b = numpy.zeros(n_degree + 2)
    
    # The last column of matrix A is for the error term 'E'.
    # It alternates in sign.
    E_array = numpy.array([(-1)**i for i in range(n_degree + 2)])
    A[:, n_degree + 1] = E_array
    
    # Track the best coefficients and minimum error found so far
    best_coeffs = None
    min_max_error = float('inf')
    
    print(f"Starting Remez algorithm for degree {n_degree}...")

    for iteration in range(max_iter):
        
        # Build the linear system A*params = b
        for i in range(n_degree + 2):
            # Powers of x for the polynomial part
            for j in range(n_degree + 1):
                A[i, j] = x_points[i]**j
            b[i] = func(x_points[i])
        
        # Solve the system for polynomial coefficients and the error E
        try:
            params = numpy.linalg.solve(A, b)
        except numpy.linalg.LinAlgError:
            print("Error: Singular matrix. Could not solve the system.")
            return None
            
        coeffs = params[:-1] # Polynomial coefficients
        E = params[-1]      # The error term
        
        # Define the residual (error) function: r(x) = F(x) - P(x)
        p = numpy.poly1d(numpy.flip(coeffs)) # Create a polynomial object
        r_i = lambda x: func(x) - p(x)
        
        # --- Find new extrema for the next iteration ---

        # 1. Bracket the roots of the residual function
        # The roots will be between the current x_points (which are sorted descending)
        interval_list = [[x_points[i+1], x_points[i]] for i in range(len(x_points)-1)]
        roots = [bisection_search(r_i, *i) for i in interval_list]
        
        # 2. Create new brackets for the extrema. These brackets must be sorted
        # and include the domain endpoints.
        sorted_roots = sorted(roots)
        extrema_brackets = [-1.0] + sorted_roots + [1.0]

        # 3. Find the new extrema within each bracketed interval. This yields n_degree + 2 points.
        extremums = [concave_max(r_i, extrema_brackets[i], extrema_brackets[i+1]) for i in range(len(extrema_brackets)-1)]
        
        # Update our set K with the new extrema points, sorted in descending order
        x_points = numpy.array(sorted(extremums, reverse=True))

        # Check for termination
        errors = numpy.abs([r_i(i) for i in x_points])
        max_error = numpy.max(errors)
        min_error = numpy.min(errors)

        # Check if this iteration produced a better result than any previous one
        if max_error < min_max_error:
            min_max_error = max_error
            best_coeffs = coeffs.copy() # Store a copy of the best coefficients

        print(f"Iter {iteration+1:2d}: Max Error = {max_error:.6f}, Min Error = {min_error:.6f}, Best Error = {min_max_error:.6f}")
        
        if (max_error - min_error) < 1e-6 * max_error:
            print("\nConverged!")
            break
            
    # Return the best coefficients found across all iterations
    return numpy.flip(best_coeffs)

def main():
    try:
        n_str = input("Enter the polynomial degree (e.g., 3): ")
        n = int(n_str)
        if n < 0:
            print("Degree must be a non-negative integer.")
            return
            
        coeffs = remez(func, n)
        
        if coeffs is not None:
            print("\nOptimal Polynomial Coefficients (highest degree first):")
            print(coeffs)
            
            # For verification, create the polynomial and check the error
            p_final = numpy.poly1d(coeffs)
            x_test = numpy.linspace(-1, 1, 200)
            error_final = func(x_test) - p_final(x_test)
            print(f"\nFinal Max Error on a fine grid: {numpy.max(numpy.abs(error_final)):.6f}")

    except ValueError:
        print("Invalid input. Please enter an integer.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()

