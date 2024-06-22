def factorial_calc(num):
    if num == 0 or num == 1:
        return 1
    else:
        result = 1
        for i in range(2, num + 1):
            result = result * i 
        return result
    
## def sin cos sinh sosh:
def apporx_sin(x ,n):
    result = 0
    for i in range (n):
        recepie = ((-1) ** i) * (x ** (2 * i + 1 )) / factorial_calc(2 * i + 1)
        result = result + recepie
    return result

def approx_cos(x, n):
    result = 0 
    for i in range(n):
        recepie = ((-1) ** i ) * (x ** (2 * i)) / factorial_calc(2*i)
        result = result + recepie
    return result

def approx_sinh(x, n):
    result = 0
    for i in range(n):
        recepie = (x ** (2 * i + 1)) / factorial_calc(2 * i + 1)
        result = result + recepie
    return result

def approx_cosh(x, n):
    result = 0
    for i in range(n):
        recepie = (x ** (2 * i)) / factorial_calc( 2 * i)
        result = result + recepie
    return result

x = 3.14
n = 10

sin_approx = apporx_sin(x, n)
print(f"approx_sin(x={x}, n={n}) = {sin_approx}")
cos_approx = approx_cos(x, n)
print(f"approx_cos(x={x}, n={n}) = {cos_approx}")
sinh_approx = approx_sinh(x, n)
print(f"approx_sinh(x={x}, n={n}) = {sinh_approx}")
cosh_approx = approx_cosh(x, n)
print(f"approx_cosh(x={x}, n={n}) = {cosh_approx}")


    