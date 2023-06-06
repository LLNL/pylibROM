import sys
sys.path.append("..")

import build.pylibROM.linalg as libROM
import numpy as np


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

#Multiplies two matrices element-wise
result_matrix=m1.elementwise_mult(m1)
print("Result matrix of Element wise Matrix multiplication of m1 with m1",result_matrix.get_data())

#Multiplies two matrices element-wise and fills result with the answer
m3=libROM.Matrix(4,5, False,True)
m3.fill(2.0)
m3.elementwise_mult(m1,m3)
print("Element wise Matrix multiplication of m3 with m1",m3.get_data())

#Square every element in the matrix
result_matrix = m1.elementwise_square()
print("Result matrix of element wise Square multiplication of matrix m1 ",result_matrix.get_data())

#Square every element in the matrix
m1.elementwise_square(m1)
print("Square every element in the matrix m1 ",m1.get_data())

#Computes a += this*b*c
a=libROM.Vector(4,False)
a.fill(2.0)
b=libROM.Vector(5,False)
b.fill(3.0)
c=3.0
m3.multPlus(a,b,c)
print("multPlus function of m3",a.get_data())

# Multiplies the transpose of a Matrix with other and returns the product
m1=m2
result_matrix1 = m1.transposeMult(m2)
print("Matrix multiplication of transpose of m1 and m2",result_matrix1.get_data())

# Multiplies the transpose of a Matrix with other and returns the product
result_matrix2 =  libROM.Matrix()
m2.transposeMult(m1,result_matrix2)
print("The product Matrix of transpose multiplication of m2 and m1",result_matrix2.get_data())


# Multiplies the transpose of a Matrix with other vector and returns the product
m1 = libROM.Matrix(3, 4,False,False)
v2 = libROM.Vector(3, False)
m1.fill(2.0)
v2.fill(2.0)
result_vector1 = m1.transposeMult(v2)
print("Matrix multiplication of transpose of m1 and vector v2",result_vector1.get_data())

# Multiplies the transpose of a Matrix with other vector and returns the product
result_vector2 =  libROM.Vector()
m1.transposeMult(v2,result_vector2) 
print("The product vector of transpose multiplication of m1 and vector v2",result_vector2.get_data())

#Computes and returns the inverse of this
m2 = libROM.Matrix(2,2,False,False)
m2.fill(3.0)
m2.__setitem__(0, 0,5.0) 
m2.__setitem__(0, 1,8.0) 
result_matrix1 = m2.inverse()
print("Inverse of matrix m2 (first overload):")
print(result_matrix1.get_data())

# Compute and store the inverse of m1 in the result_matrix using the second overload
result_matrix2 = libROM.Matrix(2,2,False,False)
m2.inverse(result_matrix2)
print("Result matrix of inverse of matrix m2 (second overload):")
print(result_matrix2.get_data())

# Compute the inverse of m1 and store it in m1 itself using the third overload
m2 = libROM.Matrix(2,2,False,False)
m2.fill(3.0)
m2.__setitem__(0, 0,5.0) 
m2.__setitem__(0, 1,8.0) 
m2.inverse()
print("Matrix m2 after inverting itself (third overload):")
print(m2.get_data())

# Get a column as a Vector
column = m1.getColumn(1)
print("column 1 of matrix m1:", column.get_data())  

#Get a column as a vector
result_vector1=libROM.Vector()
m1.getColumn(1,result_vector1)
print("column 1 of matrix m1 as a vector",result_vector1.get_data())

# Transpose the matrix
print("matrix m2")
for i in range(m2.numRows()):
    for j in range(m2.numColumns()):
        print(m2(i, j), end=" ")
    print()
m2.transpose()
print("matrix m2 after transpose")
# Print the transposed matrix
for i in range(m2.numRows()):
    for j in range(m2.numColumns()):
        print(m2(i, j), end=" ")
    print()

# Print the transposePseudoinverse matrix
m2.transposePseudoinverse()
print("matrix m2 after transposePseudoinverse")
for i in range(m2.numRows()):
    for j in range(m2.numColumns()):
        print(m2(i, j), end=" ")
    print()

# Apply qr_factorize to the matrix
m2 = libROM.Matrix(2,2,False,False)
m2.fill(3.0)
m2.__setitem__(0, 0,5.0) 
m2.__setitem__(0, 1,8.0) 
# result = m2.qr_factorize()

# Print the resulting matrix
# for i in range(result.numRows()):
#     for j in range(result.numColumns()):
#         print(result(i, j), end=" ")
#     print()


# Apply qrcp_pivots_transpose to the matrix
row_pivots = [1,2]
row_pivot_owner = [3,2]

# Call the qrcp_pivots_transpose function
row_pivots,row_pivot_owner=m2.qrcp_pivots_transpose(row_pivots,row_pivot_owner,1)
print("qrcp_pivots_transpose to the matrix m2",m2.get_data())
print("Row Pivots:", row_pivots)
print("row_pivot_owner:", row_pivot_owner)

# Apply orthogonalize to the matrix
m2.orthogonalize()
print("orthogonalize to the matrix m2")
for i in range(m2.numRows()):
    for j in range(m2.numColumns()):
        print(m2(i, j), end=" ")
    print()

# Set and get values using __setitem__ and __getitem__
matrix = libROM.Matrix(3, 3,False,False)
matrix.fill(3.0)
matrix.__setitem__(0, 0,2.0) 
print("Set Item (0,0) of matrix to 2.0 ",matrix.get_data())
print("Get Item (0,0)",matrix.__getitem__(0, 0) )
value= matrix(0, 0)
print("value",value)
print("call function",matrix.__call__(2,0))

ptr=m1.getData()
print("The storage for the Matrix's values on this processor",ptr)

a=libROM.Vector(4,False)
a.fill(2.0)
b=libROM.Vector(5,False)
b.fill(3.0)
result_matrix=libROM.outerProduct(a,b)
print("outerProduct",result_matrix.get_data())

result_matrix=libROM.DiagonalMatrixFactory(a)
print("DiagonalMatrixFactory",result_matrix.get_data())

result_matrix=libROM.IdentityMatrixFactory(a)
print("IdentityMatrixFactory",result_matrix.get_data())



m1=libROM.Matrix(2,2,False,False)
m1.__setitem__(0, 0, 4)
m1.__setitem__(0, 1, 0)
m1.__setitem__(1, 0, 3)
m1.__setitem__(1, 1, -5)

serialsvd1=libROM.SerialSVD(m1)
print("U",serialsvd1.U.get_data())
print("S",serialsvd1.S.get_data())
print("V",serialsvd1.V.get_data())


U=libROM.Matrix(2,2,False,False)
S=libROM.Vector(2,False)
V=libROM.Matrix(2,2,False,False)
libROM.SerialSVD(m1,U,S,V)
print("U",U.get_data())
print("S",S.get_data())
print("V",V.get_data())


m1=libROM.Matrix(3,3,False,False)
m1.__setitem__(0,0,2.0)
m1.__setitem__(0,1,-1.0)
m1.__setitem__(0,2,0.0)
m1.__setitem__(1,0,-1.0)
m1.__setitem__(1,1,3.0)
m1.__setitem__(1,2,2.0)
m1.__setitem__(2,0,0.0)
m1.__setitem__(2,1,2.0)
m1.__setitem__(2,2,4.0)
eigenpair=libROM.SymmetricRightEigenSolve(m1)
print("Eigen pair ev",eigenpair.ev.get_data())
print("Eigen pair eigs",eigenpair.eigs)

complexeigenpair=libROM.NonSymmetricRightEigenSolve(m1)
print("Complex Eigen pair ev",complexeigenpair.ev_real.get_data())
print("Complex Eigen pair ev_imaginary",complexeigenpair.ev_imaginary.get_data())
print("Complex Eigen pair eigs",complexeigenpair.eigs)

# Create the input arguments
As = libROM.Matrix(2,2,True,False)
As.fill(3.0)
At = libROM.Matrix(2,2,True,False)
At.fill(2.0)
Bs = libROM.Matrix(2,2,True,False)
Bs.fill(6.0)
Bt = libROM.Matrix(2,2,True,False)
Bt.fill(6.0)


# Call the SpaceTimeProduct function
m2 = libROM.Matrix(2,2,True,False)
m2.fill(3.0)
m1 = libROM.Matrix(2,2,True,False)
m1.fill(5.0)
v= libROM.Vector()
# result = libROM.SpaceTimeProduct(As, At, Bs, Bt)
result = libROM.SpaceTimeProduct(m1, m2, m1, m2,[1.0],False,False,False,True)
print("SpaceTimeProduct",result.get_data())













