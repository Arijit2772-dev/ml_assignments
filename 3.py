# q3 (a)


import numpy as np


array_a = np.array([10, 52, 62, 16, 16, 54, 453])  

print("--- Solutions for Q3(a) ---")
print(f"Original Array: {array_a}")

# i. Sorted array
sorted_array = np.sort(array_a)
print(f"\ni. Sorted array: {sorted_array}")

# ii. Indices of sorted array
sorted_indices = np.argsort(array_a)
print(f"ii. Indices of sorted array: {sorted_indices}")

# iii. 4 smallest elements
four_smallest = np.sort(array_a)[:4]
print(f"iii. 4 smallest elements: {four_smallest}")

# iv. 5 largest elements
five_largest = np.sort(array_a)[-5:]
print(f"iv. 5 largest elements: {five_largest}")



# q3 (b)



array_b = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0]) 

print("--- Solutions for Q3(b) ---")
print(f"Original Array: {array_b}")

# i. Integer elements only
# An element is an integer if its value is equal to its floor value.
integer_mask = (array_b == np.floor(array_b))
integer_elements = array_b[integer_mask]
print(f"\ni. Integer elements only: {integer_elements}")

# ii. Float elements only (i.e., numbers with a fractional part)
# An element has a fractional part if its value is NOT equal to its floor value.
float_mask = (array_b != np.floor(array_b))
float_elements = array_b[float_mask]
print(f"ii. Float elements only: {float_elements}")








