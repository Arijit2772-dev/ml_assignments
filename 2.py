# 2 (a)

import numpy as np

#The array
array = np.array([[1, 2, 3], [-4, 5, -6]]) #

print("--- Solutions for Q2(a) ---")
print(f"Original Array:\n{array}")

# i. Find element-wise absolute value
abs_array = np.absolute(array)
print(f"\ni. Element-wise absolute value:\n{abs_array}")

# ii. Find the 25th, 50th, and 75th percentile
print("\n--- Part (a)(ii) Percentiles ---")
percentiles_to_find = [25, 50, 75]

# For flattened array
p_flat = np.percentile(array, percentiles_to_find)
print(f"Percentiles of flattened array (25th, 50th, 75th): {p_flat}")
  #<-- Row 0 = 25th percentile of each column
  #<-- Row 1 = 50th percentile of each column
 #<-- Row 2 = 75th percentile of each column

# For each column (axis=0)
p_cols = np.percentile(array, percentiles_to_find, axis=0)
print(f"\nPercentiles for each column:\n{p_cols}")

# For each row (axis=1)
p_rows = np.percentile(array, percentiles_to_find, axis=1)
print(f"\nPercentiles for each row:\n{p_rows}")


# iii. Mean, Median and Standard Deviation
print("\n--- Part (a)(iii) Mean, Median, Standard Deviation ---")

# For flattened array
mean_flat = np.mean(array)
median_flat = np.median(array)
std_flat = np.std(array)
print(f"\nStats for flattened array:")
print(f"  Mean: {mean_flat:.4f}, Median: {median_flat:.4f}, Std Dev: {std_flat:.4f}")

# For each column (axis=0)
mean_cols = np.mean(array, axis=0)
median_cols = np.median(array, axis=0)
std_cols = np.std(array, axis=0)
print(f"\nStats for each column:")
print(f"  Means: {mean_cols}")
print(f"  Medians: {median_cols}")
print(f"  Std Devs: {std_cols}")

# For each row (axis=1)
mean_rows = np.mean(array, axis=1)
median_rows = np.median(array, axis=1)
std_rows = np.std(array, axis=1)
print(f"\nStats for each row:")
print(f"  Means: {mean_rows}")
print(f"  Medians: {median_rows}")
print(f"  Std Devs: {std_rows}")



# 2 (b)



# The array given in the assignment
a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0]) # 

print("--- Solutions for Q2(b) ---")
print(f"Original array: {a}")

# np.floor(): Rounds down to the nearest integer.
floor_vals = np.floor(a)
print(f"Floor values:   {floor_vals}")

# np.ceil(): Rounds up to the nearest integer.
ceil_vals = np.ceil(a)
print(f"Ceiling values: {ceil_vals}")

# np.trunc(): Chops off the decimal, rounding toward zero.
trunc_vals = np.trunc(a)
print(f"Truncated values: {trunc_vals}")

# np.round(): Rounds to the nearest integer (halfway values to the nearest even integer).
rounded_vals = np.round(a)
print(f"Rounded values:   {rounded_vals}")









