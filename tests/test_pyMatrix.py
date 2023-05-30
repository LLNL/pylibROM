import sys
sys.path.append("..")

import build.pylibROM.linalg as libROM


# Create two Matrix objects
m1 = libROM.Matrix(3,4, False,True)
m2 = libROM.Matrix(3,4, False, False)

m1.fill(2.0)
m2.__assign__(m1)

# Print the initial data for m1 and m2
print("Initial data for m1:", m1.get_data())
print("Initial data for m2:", m2.get_data())

# Use the addition operator
m1 += m2

# Print the updated data for v1
print("Updated data for m1 after addition:", m1.get_data())

#set size 
m1.setSize(4,5)

print("Is m1 distributed:", m1.distributed())

print("Is m1 balanced:", m1.balanced())
print("Is m2 balanced:", m2.balanced())

print("number of Rows in m1:", m1.numRows())

print("number of columns in m1:", m1.numColumns())

print("number of Distributed Rows in m1:", m1.numDistributedRows())

m1.fill(3.0)
print("Initial data for m1:", m1.get_data())

# Call the getFirstNColumns method with return type Matrix*
result1 = m1.getFirstNColumns(3)
print("Get First 3 columns of m1",result1.get_data())  

# Call the getFirstNColumns method with input parameters of n,matrix*
m3 = libROM.Matrix()
m1.getFirstNColumns(3, m3)
print("Get first 3 columns of m1 to m3",m3.get_data()) 

# Multiply matrix1 with matrix2
result_matrix1 = m2.mult(m1)
print("Matrix multiplication of m1 and m2",result_matrix1.get_data())

# Multiply matrix1 with matrix2 
result_matrix2 =  libROM.Matrix()
m2.mult(m1,result_matrix2)
print("The product Matrix of m1 and m2",result_matrix2.get_data())


# Multiply matrix1 with vector2
v1 = libROM.Vector(5, False)
v1.fill(2.0)
result_vector1 = m1.mult(v1)
print("Matrix multiplication of m1 and vector v1",result_vector1.get_data())

# Multiply matrix1 with matrix2 
result_vector2 =  libROM.Vector()
m1.mult(v1,result_vector2)
print("The product vector of m1 and vector v1",result_vector2.get_data())

# Test the first pointwise_mult function
v2 = libROM.Vector(5, False)
v2.fill(2.0)
result_vector3 =  libROM.Vector(5,False)
m1.pointwise_mult(1, v2, result_vector3)
print("The result vector of  pointwise_multiplication of 2nd row of m1 and vector v1", result_vector3.get_data())

# Test the second pointwise_mult function
m1.pointwise_mult(1, v1)
print("The product vector v1 after pointwise_multiplication with 2nd row of m1", v1.get_data())







