import numpy as np
import sympy as sp
import warnings
import math

np.set_printoptions(suppress=True)
x = sp.symbols('x')

warnings.filterwarnings("error")
flag = True


def Menu():
    print("*****************MAIN MENU*****************")

    choice = input("""
    1: PART 1 - BINARY and DECIMAL CONVERSION

    2: PART 2 - MATRIX MANIPULATION

    3: PART 3 - EIGAN VECTORS AND NORMS

    4: PART 4 - NUMERICAL ANALYSIS

    6: Quit/Log Out

    Please enter your choice: """)

    if (choice == '1'):
        ans = 0
        while (ans != -1):
            choice1 = input("""
            A: Convert Decimal to Binary
            B: Convert Binary to Decimal
            M: Back to main menu
            ENTER YOUR CHOICE: """).lower()

            if (choice1 == 'a'):
                n1 = input("ENTER A DECIMAL NUMBER TO CONVERT TO BINARY: ")
                try:
                    val = int(n1)
                    print(decimalToBinary(int(n1)))

                except ValueError:
                    try:
                        val = float(n1)
                        print(FloatDecimalToBinary(float(n1)))

                    except ValueError:
                        print("No.. input is not a number. It's a string")


            elif (choice1 == 'b'):
                n2 = input("ENTER A BINARY NUMBER TO CONVERT INTO DECIMAL: ")
                try:
                    val = int(n2)
                    print(binaryToDecimal(n2))

                except ValueError:
                    try:
                        val = float(n2)
                        print(FloatbinaryToDecimal(n2, len(n2)))

                    except ValueError:
                        print("No.. input is not a number. It's a string")


            elif (choice1 == 'm'):
                break

    elif choice == '2':
        ans = 0
        while (ans != -1):
            choice1 = input("""
                    A: Matrix Addition
                    B: Multiply
                    C: Gaussian Elimination
                    D: Gauss Jordon Elimination
                    E: Determinant Calculator
                    F: Inverse of the matrix
                    M: Back to main menu
                    ENTER YOUR CHOICE: """).lower()

            if choice1 == 'a':
                M1 = []
                M2 = []

                print("Enter the first matrix")
                M1 = Matrix(M1)
                print(M1)

                print("Enter the second matrix:-")
                M2 = Matrix(M2)
                print(M2)
                a = np.array(M1)
                b = np.array(M2)
                # if len(M1)*len(M1[0]) == len(M2)*len(M2[0]):
                if M1.shape == M2.shape:
                    print("The result is :- ")
                    print(a + b)

                else:
                    print("SORRY! The dimensions of matrix need to be EQUAL for addition")


            elif choice1 == 'b':
                M1 = []
                M2 = []

                print("Enter the first matrix")
                M1 = Matrix(M1)
                print(M1)
                print("Enter the second matrix:-")
                M2 = Matrix(M2)
                print(M2)

                a = np.array(M1)
                b = np.array(M2)

                try:
                    c=a.dot(b)
                    print("The result is :- \n ", c)
                    print("dimensions are ", c.shape)


                except ValueError:
                    print("DIMENSION ERROR! The column of matrix 1 should be = to the row of matrix 2")


            elif (choice1 == 'c'):
                a = []
                b = []
                print("Enter the matrix for the equations NOT including the constants after '=' ")
                a = Matrix(a)
                print(a)
                print("Enter the vector matrix (Form n*1)")
                b = Matrix(b)
                print(b)
                try:
                    X = GaussE(a, b)
                    print("The resultant matrix is \n ", X)

                except RuntimeWarning:
                    print("Infinitely many solutions or no solutions")

            elif (choice1 == 'd'):
                M1 = []
                M2 = []
                print("Enter the equations as matrix")
                M1 = Matrix(M1)
                print(M1)
                print("Enter the vector matrix (Form N*1)")
                M2 = Matrix(M2)
                print(M2)
                A, X = gaussJ(M1, M2)
                print("The transformed matrix is")
                print(A)
                print("The solution is")
                print(X)

            elif choice1 == 'e':
                M = []
                print("Enter N*N matrix to calculate the Determinant")
                M = Matrix(M)
                l = list(M)
                print("The determinant of \n",M, " is ", det(l))

            elif choice1 == 'f':
                M = []
                print("Enter N*N matrix to calculate the inverse")
                M = Matrix(M)
                print(M)
                a=np.array(M)
                print("The determinant is ", det(list(M)))
                if det(list(M))==0:
                    print("There is no inverse as determinant is 0")
                else:
                    print("The inverse matrix is ")
                    print(inverse(M))

            elif choice1 =='g':
                M=[]
                M= Matrix(M)
                print(comatrix(M))


            elif choice1 == 'm':
                break

            else:
                print("Invalid Choice, Try again")

    elif choice == '3':
        ans = 0
        while ans != -1:
            choice1 = input("""
                                   A: Eigenvalues and EigenVectors
                                   B: Matrix and Vector Norms
                                   m: Back to main Menu
                                   ENTER YOUR CHOICE: """).lower()
            if(choice1=='a'):

                print("Enter the Matrix to calculate eigenvalues and eigenvectors")
                M=[]
                M=Matrix(M)
                a = np.array(M)
                vals, vec = np.linalg.eig(a)
                np.set_printoptions(suppress=True)
                print("The Eigenvalues for given Matrix is ", vals)
                print("The Eigenvectors for given Matrix is \n", vec)

            elif choice1=='b':
                print("Enter a vector of 3*1 dimension")
                M=[]
                M=Matrix(M)
                sumE=0
                sumT=0
                for x in M:
                    sumE+=x**2
                    sumT+=abs(x)
                print("The Euclidean Norm is ", math.sqrt(sumE) )
                print("The Taxicab Norm is ", sumT)
                print("The Maximum Norm is ", max(map(abs,M)))


            elif choice1 == 'm':
                break

            else :
                print("Invalid Choice, Try again")



    elif choice =='4':
        ans = 0
        while (ans != -1):
            choice1 = input("""
                           A: Bisection method
                           B: Newton Method
                           C: Horner's Method
                           m: Back to main Menu
                           ENTER YOUR CHOICE: """).lower()
            if choice1 =='a':
                print("Enter the interval a, b and tolerance level and funtion to find out the root.")

                try:
                    a = eval(input("Enter a = "))
                    b = eval(input("Enter b = "))
                    tol = eval(input("Enter tolerance level "))
                    print("for function example: 0.5*(np.cos(x)-x**2)")
                    bisection(a,b,tol)
                except (NameError,TypeError) :
                    print("You may have entered string values or not use np functions in right way")

            elif choice1 =='b':

                try:
                    newtons_method()
                except (NameError, TypeError,ValueError,AttributeError):
                    print("You may have entered string values or not have written the functions in the right way")

            elif choice1 =='c':
                n = int(input("Enter the number of elements: "))
                # Below line read inputs from user using map() function
                poly = list(map(int, input("\nEnter the coefficients of the polynomial starting from lowest degree("
                                           "space seperated) : ").strip().split()))[:n]
                x = float(input("\nEnter the value you want to evaluate the polynomial for "))
                ans, count = poly_horner(poly, x)
                print("Value of polynomial is ", ans)
                print("The number of addition and multiplications = ", count)

            elif choice1=='m':
                break

            else :
                print("Invalid Choice, Try again")



    elif choice == '6':
        exit()

    else :
        print("Invalid choice, Try again")

def decimalToBinary(n):
    return bin(n).replace("0b", "")


def FloatDecimalToBinary(num):
    binary = ""

    # Fetch the integral part of
    # decimal number
    Integral = int(num)

    # Fetch the fractional part
    # decimal number
    fractional = num - Integral

    # Conversion of integral part to
    # binary equivalent
    while (Integral):
        rem = Integral % 2

        # Append 0 in binary
        binary += str(rem)

        Integral //= 2

    # Reverse string to get original
    # binary equivalent
    binary = binary[:: -1]

    # Append point before conversion
    # of fractional part
    binary += '.'

    # Conversion of fractional part
    # to binary equivalent
    while (fractional):

        # Find next bit in fraction
        fractional *= 2
        fract_bit = int(fractional)

        if (fract_bit == 1):

            fractional -= fract_bit
            binary += '1'

        else:
            binary += '0'


    return binary


def binaryToDecimal(n):
    return int(n, 2)


def FloatbinaryToDecimal(binary, length):
    # Fetch the radix point
    point = binary.find('.')

    # Update point if not found
    if (point == -1):
        point = length

    intDecimal = 0
    fracDecimal = 0
    twos = 1

    # Convert integral part of binary
    # to decimal equivalent
    for i in range(point - 1, -1, -1):
        # Subtract '0' to convert
        # character into integer
        intDecimal += ((ord(binary[i]) -
                        ord('0')) * twos)
        twos *= 2

    # Convert fractional part of binary
    # to decimal equivalent
    twos = 2

    for i in range(point + 1, length):
        fracDecimal += ((ord(binary[i]) -
                         ord('0')) / twos)
        twos *= 2.0

    # Add both integral and fractional part
    x = intDecimal + fracDecimal

    return x

# Function to take input from user and put it matrix like array
def Matrix(m):
        R = int(input("Enter the number of rows: "))
        C = int(input("Enter the number of columns: "))
        print("Enter the entries in a single line row wise (separated by space): ")
        entries = list(map(float, input().split()))
        try:
            m = np.array(entries).reshape(R, C)
            return m
        except ValueError:
            print("You probably didn't fill in correct values for you matrix, please try again!")
            main()

def GaussE(a, b):
    n = len(b)
    x = np.zeros(n, float)
    # ELIMINATION

    for k in range(n - 1):
        # Checks if pivoting needs to be done and changes row if yes
        if np.fabs(a[k, k]) < 1.0e-12:
            for i in range(k + 1, n):
                if np.fabs(a[i, k]) > np.fabs(a[k, k]):
                    a[[k, i]] = a[[i, k]]
                    b[[k, i]] = b[[i, k]]
                    break
        # Loops till the last element and then sees the factor and does Gauss elimination for matrix a and b
        for i in range(k + 1, n):
            if a[i, k] == 0: continue
            factor = a[k, k] / a[i, k]
            for j in range(k, n):
                a[i, j] = a[k, j] - a[i, j] * factor
            b[i] = b[k] - b[i] * factor
    # We now have matrix in triangular form, we will back-substitute
    x[n - 1] = b[n - 1] / a[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        sum_ax = 0
        for j in range(i + 1, n):
            sum_ax += a[i, j] * x[j]
        x[i] = (b[i] - sum_ax) / a[i, i]
    return x


def gaussJ(a, b):
    n = len(b)
    for k in range(n):
        # Partial Pivoting
        if np.fabs(a[k, k]) < 1.0e-12:
            for i in range(k + 1, n):
                if np.fabs(a[i, k]) > np.fabs(a[k, k]):
                    for j in range(k, n):
                        a[k, j], a[i, j] = a[i, j], a[k, j]
                    b[[k, i]] = b[[i, k]]
                    break

        # Division of pivot row

        pivot = a[k, k]
        for j in range(k, n):
            a[k, j] /= pivot
        b[k] /= pivot

        # Elimination loop

        for i in range(n):
            if i == k or a[i, k] == 0: continue
            factor = a[i, k]
            for j in range(k, n):
                a[i, j] -= factor * a[k, j]
            b[i] -= factor * b[k]
    # Returns Transformed Matrix a and the solution matrix b
    return a, b


def det(l):
    # Uses recursive function calls to calculate determinant
    n = len(l)
    # Calculates determinant for matrix of 3*3 and above
    if (n > 2):
        i = 1
        t = 0
        sum = 0
        while t <= n - 1:
            d = {}
            t1 = 1
            while t1 <= n - 1:
                m = 0
                d[t1] = []
                while m <= n - 1:
                    if (m == t):
                        u = 0
                    else:
                        d[t1].append(l[t1][m])
                    m += 1
                t1 += 1
            l1 = [d[x] for x in d]
            sum = sum + i * (l[0][t]) * (det(l1))
            i = i * (-1)
            t += 1
        return sum
    # Calculates determinant for 2*2 matrix (ad-bc)
    else:
        return (l[0][0] * l[1][1] - l[0][1] * l[1][0])

def minor(A, i, j):
    m = []
    rows = len(A)
    cols = rows

    for r in range(rows):
        l = []
        for c in range(cols):
            if c != j:
                l.append(A[r][c])
        if r != i:
            m.append(l)
    return np.linalg.det(m)


def comatrix(A):
    rows = len(A)
    cols = rows
    c = []
    for i in range(rows):
        l = []
        for j in range(cols):
            l.append((-1) ** (i + j) * minor(A, i, j))
        c.append(l)
    return (c)


def adjunct(A):
    return np.transpose(comatrix(A))


def inverse(A):

    return (1 / np.linalg.det(A) * adjunct(A))

def func(expr, x):
    return eval(expr)

def bisection(a, b,tol):
    inpV = input("Input your function here (use np.cos(X) and np.sin(x) for trig values and np.exp(x) for exponential values: ")
    if (func(inpV, a) * func(inpV, b) >= 0):
        print("You have not assumed right a and b. At one of the intervals, function should evluate to 0 \n")
        return

    c = a
    while (abs(func(inpV,c)) >= tol):

        # Find middle point
        c = (a + b) / 2

        # Check if middle point is root
        if (func(inpV, c) == 0.0):
            break

        # Decide the side to repeat the steps
        if (func(inpV, c) * func(inpV, a) < 0):
            b = c
        else:
            a = c

    print("The value of root is : ", "%.4f" % c)


def f(symx):
    tmp = sp.sympify(symx)
    return tmp

# Function to calculate derivatives
def fdiff(symx):
    tmp = sp.diff(f(symx))
    return tmp

def newtons_method():
    guess = sp.sympify(float(input("Enter an initial guess (Please convert your starting guess as integer or float): "))) # Convert to an int immediately.
    symx = input("Input your function here (use cos(5*X) and sin(x) for trig values and exp(x) for exponential values: ")
    div = f(symx)/fdiff(symx)
    # Here I'm using The Newton Method Algorithm for max of 10 iterations
    for i in range(1, 10):
        print(i-1," Iteration   ",guess.evalf())
        nextGuess = guess - div.subs(x, guess)
        guess = nextGuess

def poly_horner(A, x):
    p = A[-1]
    i = len(A) - 2
    count = 0
    while i >= 0:
        p = p * x + A[i]
        i -= 1
        count += 1
    return p, count

def main():
    print("********************************************")
    print("WELCOME TO THE MATH 225 COURSE - Numerical Analysis and Linear Algebra for CS !!")
    print("THIS IS A TOOL TO SOLVE ALL THE THINGS THAT I LEARNT IN THE COURSE")
    print("********************************************")
    while (flag == True):
        Menu()

main()
