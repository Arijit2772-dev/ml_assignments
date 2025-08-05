# reverse the numpy array ...
# reverse the numpy array  = np.array([1, 2, 3, 6, 4, 5])

# part(a)
import numpy as np

arr = np.array([1, 2, 3, 6, 4, 5]) 

reversed_arr = arr[::-1]

print("Original array:", arr)
print("Reversed array:", reversed_arr)


#part(b)

array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]]) # [cite: 4]

print("Original array:")
print(array1)

# Method 1: Using the flatten() method
# object method....
flattened_arr_1 = array1.flatten()
print("\nMethod 1 - Using flatten(): --> ", flattened_arr_1)

# Method 2: Using the ravel() function
# standlone funciton...
flattened_arr_2 = np.ravel(array1)
print("Method 2 - Using ravel(): --> ", flattened_arr_2)



#part(c)
import numpy as np

arr1 = np.array([[1, 2], [3, 4]]) 
arr2 = np.array([[1, 2], [3, 4]])  

print("arr1:")
print(arr1)
print("arr2:")
print(arr2)

# 1. Element-wise comparison
   # ( returns a boolean array... )
element_wise_comparison = (arr1 == arr2)
print("Result of element-wise comparison:")
print(element_wise_comparison)

# 2. complete array equal or not..
are_equal = np.array_equal(arr1, arr2)
print("Are the arrays completely equal? ,{are_equal}")





#part (d)
# (i)


import numpy as np


# made array..
x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1]) 



# array returned..
#bincount returns an array from 0 to .. highest numer+1 of their frequeincies..
counts_x = np.bincount(x)

# index ..
# index is same as number here.. because of bincount...
most_frequent_value = np.argmax(counts_x)


# boolean array created...
# ( only single value...)
condition = (x == most_frequent_value)


#tuple returned..
indices_x = np.where(condition)[0]

print("For array x:",x)
print("The array is: ",x)
print("The most frequent value is:",most_frequent_value)
print("Its indices are: ",indices_x)



# for second part 
# what is the numerical value of the highest count?"

# (ii)

import numpy as np

# --- Solution for Part d(i) ---

# [cite_start]Define the first array [cite: 11]
x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1]) 

# Use bincount and argmax, which works here since there are no ties
counts_x = np.bincount(x)
most_frequent_x = np.argmax(counts_x)
indices_x = np.where(x == most_frequent_x)[0]

print("--- Solution for Part d(i) ---")
print(f"Array x: {x}")
print(f"The most frequent value is: {most_frequent_x}")
print(f"Its indices are: {indices_x}")


# --- Solution for Part d(ii) ---

y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])


counts_y = np.bincount(y)

max_count = np.max(counts_y)


#values....
most_frequent_y = np.where(counts_y == max_count)[0]

print("\n--- Solution for Part d(ii) ---")
print(f"Array y: {y}")
print(f"The most frequent values (due to a tie) are: {most_frequent_y}")

#indices...
for value in most_frequent_y:
    indices_y = np.where(y == value)[0]
    print(f"Indices for value '{value}' are: {indices_y}")




# (e part )


# np.array is more common now, but this works perfectly.
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]') 

print("Original matrix gfg:")
print(gfg)

# i. Sum of all elements
total_sum = np.sum(gfg)
print(f"\ni. Sum of all elements: {total_sum}")

# ii. Sum of all elements row-wise (axis=1)
row_wise_sum = np.sum(gfg, axis=1)
print("\nii. Sum of all elements row-wise:")
print(row_wise_sum)

# iii. Sum of all elements column-wise (axis=0)
col_wise_sum = np.sum(gfg, axis=0)
print("\niii. Sum of all elements column-wise:")
print(col_wise_sum)




# ( f part ..)

# The matrix 
n_array = np.array([[55, 25, 15], [30, 44, 2], [11, 45, 77]]) # 

print("Original matrix n_array:")
print(n_array)

# i. Sum of diagonal elements (Trace)
trace = np.trace(n_array)
print(f" Sum of diagonal elements: {trace}")

# ii. Eigenvalues and iii. Eigenvectors
# np.linalg.eig() returns both values and vectors at once
eigenvalues, eigenvectors = np.linalg.eig(n_array)
print(f"ii. Eigenvalues: {eigenvalues}")
print(f"iii. Eigenvectors:\n{eigenvectors}")

# iv. Inverse of matrix
inverse_matrix = np.linalg.inv(n_array)
print(f"iv. Inverse of matrix: {inverse_matrix}")

# v. Determinant of matrix
determinant = np.linalg.det(n_array)
print(f" v. Determinant of matrix: {determinant}")






# ( g part )


# --- Solution for Part g(i) ---
print("--- Solution for Part g(i) ---")
p1 = np.array([[1, 2], [2, 3]]) 
q1 = np.array([[4, 5], [6, 7]]) 

# Matrix Multiplication
product_1 = p1 @ q1

# Covariance
covariance_1 = np.cov(p1, q1)

print(f"Product of p1 and q1:\n{product_1}")
print(f"\nCovariance of p1 and q1:\n{covariance_1}")


# --- Solution for Part g(ii) ---
print("\n--- Solution for Part g(ii) ---")
# The source document lists these two matrices for the second part of the question.
q2 = np.array([[4, 5, 1], [6, 7, 2]]) # [cite: 39]
p2 = np.array([[1, 2], [2, 3], [4, 5]]) # [cite: 40]

# Matrix Multiplication:
# The shape of q2 is (2, 3) and p2 is (3, 2).
# Therefore, q2 @ p2 is a valid multiplication.
product_2 = q2 @ p2

# Covariance:.

covariance_2 = np.cov(q2.flatten(), p2.flatten())

print(f"Product of q2 and p2:\n{product_2}")
print(f"\nCovariance of flattened q2 and p2:\n{covariance_2}")




# ( h part)

x = np.array([[2, 3, 4], [3, 2, 9]]) 
y = np.array([[1, 5, 0], [5, 10, 3]]) 

print("--- Solutions for Part h ---")
print(f"Matrix x:\n{x}")
print(f"Matrix y:\n{y}")

# 1. Inner Product
inner_prod = np.inner(x, y)
print(f"\nInner Product:\n{inner_prod}")

# 2. Outer Product
# np.outer requires 1D arrays, so we flatten the matrices first.
outer_prod = np.outer(x.flatten(), y.flatten())
print(f"\nOuter Product (of flattened arrays):\n{outer_prod}")

# 3. Cartesian Product (interpreted as element-wise product)
cartesian_prod = x * y
print(f"\nCartesian (element-wise) Product:\n{cartesian_prod}")







































