import pytest
import numpy as np
try:
    # import pip-installed package
    import pylibROM
    import pylibROM.linalg as libROM
    import pylibROM.utils as utils
except ModuleNotFoundError:
    # If pip-installed package is not found, import cmake-built package
    import sys
    sys.path.append("../build")
    import _pylibROM as pylibROM
    import _pylibROM.linalg as libROM
    import _pylibROM.utils as utils

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
m2.invert()
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
m2 = libROM.Matrix(4,4,False,False)

# Apply orthogonalize to the matrix
m2.fill(3.0)
m2.__setitem__(0, 0,5.0) 
m2.__setitem__(0, 1,8.0) 
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

def test_plus():
    m1 = libROM.Matrix(3,4, False,True)
    m2 = libROM.Matrix(3,4, False, False)

    m1.fill(2.0)
    m2.__assign__(m1)

    # Print the initial data for m1 and m2
    print("Initial data for m1:", m1.get_data())
    print("Initial data for m2:", m2.get_data())

    result = m2.get_data()
    assert result == [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]
    
    # Use the addition operator
    m1 += m2

    # Print the updated data for v1
    print("Updated data for m1 after addition:", m1.get_data())
    assert m1.get_data() == [[4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0]]

    #set size 
    m1.setSize(4,5)

    print("Is m1 distributed:", m1.distributed())
    assert m1.distributed() == False
    print("Is m1 balanced:", m1.balanced())
    assert m1.balanced() == True
    print("Is m2 balanced:", m2.balanced())
    assert m2.balanced() == True
    print("number of Rows in m1:", m1.numRows())
    assert m1.numRows() == 4
    print("number of columns in m1:", m1.numColumns())
    assert m1.numColumns() == 5
    print("number of Distributed Rows in m1:", m1.numDistributedRows())
    assert m1.numDistributedRows() == 4
    
    m1.fill(3.0)
    print("Initial data for m1:", m1.get_data())
    assert m1.get_data() == [[3.0, 3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0, 3.0]]
    
    # Call the getFirstNColumns method with return type Matrix*
    result1 = m1.getFirstNColumns(3)
    print("Get First 3 columns of m1",result1.get_data())  
    assert result1.get_data() == [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]] 

    # Call the getFirstNColumns method with input parameters of n,matrix*
    m3 = libROM.Matrix()
    m1.getFirstNColumns(3, m3)
    print("Get first 3 columns of m1 to m3",m3.get_data()) 
    assert result1.get_data() == [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]] 

    # Multiply matrix1 with matrix2
    result_matrix1 = m2.mult(m1)
    print("Matrix multiplication of m1 and m2",result_matrix1.get_data())
    assert result_matrix1.get_data() == [[24.0, 24.0, 24.0, 24.0, 24.0], [24.0, 24.0, 24.0, 24.0, 24.0], [24.0, 24.0, 24.0, 24.0, 24.0]]

    # Multiply matrix1 with matrix2 
    result_matrix2 =  libROM.Matrix()
    m2.mult(m1,result_matrix2)
    print("The product Matrix of m1 and m2",result_matrix2.get_data())
    assert result_matrix2.get_data() == [[24.0, 24.0, 24.0, 24.0, 24.0], [24.0, 24.0, 24.0, 24.0, 24.0], [24.0, 24.0, 24.0, 24.0, 24.0]]

    # Multiply matrix1 with vector2
    v1 = libROM.Vector(5, False)
    v1.fill(2.0)
    result_vector1 = m1.mult(v1)
    print("Matrix multiplication of m1 and vector v1",result_vector1.get_data())
    assert result_vector1.get_data() == [30.0, 30.0, 30.0, 30.0] 

    # Multiply matrix1 with matrix2 
    result_vector2 =  libROM.Vector()
    m1.mult(v1,result_vector2)
    print("The product vector of m1 and vector v1",result_vector2.get_data())
    assert result_vector2.get_data() == [30.0, 30.0, 30.0, 30.0]

    # Test the first pointwise_mult function
    v2 = libROM.Vector(5, False)
    v2.fill(2.0)
    result_vector3 =  libROM.Vector(5,False)
    m1.pointwise_mult(1, v2, result_vector3)
    print("The result vector of  pointwise_multiplication of 2nd row of m1 and vector v1", result_vector3.get_data())
    assert result_vector3.get_data() == [6.0, 6.0, 6.0, 6.0, 6.0] 

    # Test the second pointwise_mult function
    m1.pointwise_mult(1, v1)
    print("The product vector v1 after pointwise_multiplication with 2nd row of m1", v1.get_data())
    assert v1.get_data() == [6.0, 6.0, 6.0, 6.0, 6.0]

    #Multiplies two matrices element-wise
    result_matrix=m1.elementwise_mult(m1)
    print("Result matrix of Element wise Matrix multiplication of m1 with m1",result_matrix.get_data())
    assert result_matrix.get_data() == [[9.0, 9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0, 9.0]] 

    #Multiplies two matrices element-wise and fills result with the answer
    m3=libROM.Matrix(4,5, False,True)
    m3.fill(2.0)
    m3.elementwise_mult(m1,m3)
    print("Element wise Matrix multiplication of m3 with m1",m3.get_data())
    assert m3.get_data() == [[6.0, 6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 6.0, 6.0, 6.0]]

    #Square every element in the matrix
    result_matrix = m1.elementwise_square()
    print("Result matrix of element wise Square multiplication of matrix m1 ",result_matrix.get_data())
    assert result_matrix.get_data() == [[9.0, 9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0, 9.0]]

    #Square every element in the matrix
    m1.elementwise_square(m1)
    print("Square every element in the matrix m1 ",m1.get_data())
    assert m1.get_data() == [[9.0, 9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0, 9.0], [9.0, 9.0, 9.0, 9.0, 9.0]]

    #Computes a += this*b*c
    a=libROM.Vector(4,False)
    a.fill(2.0)
    b=libROM.Vector(5,False)
    b.fill(3.0)
    c=3.0
    m3.multPlus(a,b,c)
    print("multPlus function of m3",a.get_data())
    assert a.get_data() == [272.0, 272.0, 272.0, 272.0]

    # Multiplies the transpose of a Matrix with other and returns the product
    m1=m2
    result_matrix1 = m1.transposeMult(m2)
    print("Matrix multiplication of transpose of m1 and m2",result_matrix1.get_data())
    assert result_matrix1.get_data() == [[12.0, 12.0, 12.0, 12.0], [12.0, 12.0, 12.0, 12.0], [12.0, 12.0, 12.0, 12.0], [12.0, 12.0, 12.0, 12.0]]

    # Multiplies the transpose of a Matrix with other and returns the product
    result_matrix2 =  libROM.Matrix()
    m2.transposeMult(m1,result_matrix2)
    print("The product Matrix of transpose multiplication of m2 and m1",result_matrix2.get_data())
    assert result_matrix2.get_data() == [[12.0, 12.0, 12.0, 12.0], [12.0, 12.0, 12.0, 12.0], [12.0, 12.0, 12.0, 12.0], [12.0, 12.0, 12.0, 12.0]]

    # Multiplies the transpose of a Matrix with other vector and returns the product
    m1 = libROM.Matrix(3, 4,False,False)
    v2 = libROM.Vector(3, False)
    m1.fill(2.0)
    v2.fill(2.0)
    result_vector1 = m1.transposeMult(v2)
    print("Matrix multiplication of transpose of m1 and vector v2",result_vector1.get_data())
    assert result_vector1.get_data() == [12.0, 12.0, 12.0, 12.0] 

    # Multiplies the transpose of a Matrix with other vector and returns the product
    result_vector2 =  libROM.Vector()
    m1.transposeMult(v2,result_vector2) 
    print("The product vector of transpose multiplication of m1 and vector v2",result_vector2.get_data())
    assert result_vector2.get_data() == [12.0, 12.0, 12.0, 12.0]

    #Computes and returns the inverse of this
    m2 = libROM.Matrix(2,2,False,False)
    m2.fill(3.0)
    m2.__setitem__(0, 0,5.0) 
    m2.__setitem__(0, 1,8.0) 
    result_matrix1 = m2.inverse()
    print("Inverse of matrix m2 (first overload):")
    print(result_matrix1.get_data())
    assert np.allclose(result_matrix1.get_data(), [[-0.3333333333333332, 0.8888888888888886], [0.33333333333333326, -0.5555555555555554]])

    # Compute and store the inverse of m1 in the result_matrix using the second overload
    result_matrix2 = libROM.Matrix(2,2,False,False)
    m2.inverse(result_matrix2)
    print("Result matrix of inverse of matrix m2 (second overload):")
    print(result_matrix2.get_data())
    assert np.allclose(result_matrix2.get_data(), [[-0.3333333333333332, 0.8888888888888886], [0.33333333333333326, -0.5555555555555554]])

    # Get a column as a Vector
    column = m1.getColumn(1)
    print("column 1 of matrix m1:", column.get_data())  
    assert column.get_data() == [2.0, 2.0, 2.0]

    #Get a column as a vector
    result_vector1=libROM.Vector()
    m1.getColumn(1,result_vector1)
    print("column 1 of matrix m1 as a vector",result_vector1.get_data())
    assert result_vector1.get_data() == [2.0, 2.0, 2.0] 

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
    assert m2.get_data() == [[5.0, 3.0], [8.0, 3.0]] 


    # Print the transposePseudoinverse matrix
    m2.transposePseudoinverse()
    print("matrix m2 after transposePseudoinverse")
    for i in range(m2.numRows()):
        for j in range(m2.numColumns()):
            print(m2(i, j), end=" ")
        print()
    assert m2.get_data() == [[-0.3333333333333333, 0.3333333333333333], [0.8888888888888888, -0.5555555555555556]]

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

    m2 = libROM.Matrix(4,4,False,False)

    # Set and get values using __setitem__ and __getitem__
    matrix = libROM.Matrix(3, 3,False,False)
    matrix.fill(3.0)
    matrix.__setitem__(0, 0,2.0) 
    print("Set Item (0,0) of matrix to 2.0 ",matrix.get_data())
    assert matrix.get_data() == [[2.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]] 
    print("Get Item (0,0)",matrix.__getitem__(0, 0) )
    assert matrix.__getitem__(0, 0) == 2.0 
    value= matrix(0, 0) 
    print("value",value)
    assert value == 2.0 
    print("call function",matrix.__call__(2,0))
    assert matrix.__call__(2,0) == 3.0 

    # this is not supposed to be a single value!!
    ptr=m1.getData()
    print("The storage for the Matrix's values on this processor",ptr)
    assert(ptr.shape[0] == m1.numRows())
    assert(ptr.shape[1] == m1.numColumns())

    a=libROM.Vector(4,False)
    a.fill(2.0)
    b=libROM.Vector(5,False)
    b.fill(3.0)
    result_matrix=libROM.outerProduct(a,b)
    print("outerProduct",result_matrix.get_data())
    assert result_matrix.get_data() == [[6.0, 6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 6.0, 6.0, 6.0], [6.0, 6.0, 6.0, 6.0, 6.0]]  

    result_matrix=libROM.DiagonalMatrixFactory(a)
    print("DiagonalMatrixFactory",result_matrix.get_data())
    assert result_matrix.get_data() == [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0]]

    result_matrix=libROM.IdentityMatrixFactory(a)
    print("IdentityMatrixFactory",result_matrix.get_data())
    assert result_matrix.get_data() == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] 

    m1=libROM.Matrix(2,2,False,False)
    m1.__setitem__(0, 0, 4)
    m1.__setitem__(0, 1, 0)
    m1.__setitem__(1, 0, 3)
    m1.__setitem__(1, 1, -5)

    serialsvd1=libROM.SerialSVD(m1)
    print("U",serialsvd1.U.get_data())
    print("S",serialsvd1.S.get_data())
    print("V",serialsvd1.V.get_data())
    np.testing.assert_allclose(serialsvd1.U.get_data(), [[-0.7071067811865475, 0.7071067811865475], [-0.7071067811865475, -0.7071067811865475]])
    np.testing.assert_allclose(serialsvd1.S.get_data(), [6.324555320336759, 3.162277660168379])
    np.testing.assert_allclose(serialsvd1.V.get_data(), [[-0.4472135954999579, -0.8944271909999159], [-0.8944271909999159, 0.4472135954999579]])

    U=libROM.Matrix(2,2,False,False)
    S=libROM.Vector(2,False)
    V=libROM.Matrix(2,2,False,False)
    libROM.SerialSVD(m1,U,S,V)
    print("U",U.get_data())
    print("S",S.get_data())
    print("V",V.get_data())
    np.testing.assert_allclose(U.get_data(), [[-0.7071067811865475, 0.7071067811865475], [-0.7071067811865475, -0.7071067811865475]])
    np.testing.assert_allclose(S.get_data(), [6.324555320336759, 3.162277660168379])
    np.testing.assert_allclose(V.get_data(), [[-0.4472135954999579, -0.8944271909999159], [-0.8944271909999159, 0.4472135954999579]])


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
    np.testing.assert_allclose(eigenpair.ev.get_data(), [[0.5932333119173844, 0.7864356987513784, -0.17202653679290808], [0.6793130619863368, -0.3743619547830712, 0.6311789687764829], [-0.4319814827585531, 0.4912962635115681, 0.756320024865991]])
    np.testing.assert_allclose(eigenpair.eigs, [0.8548973087995777, 2.4760236029181337, 5.669079088282289])
    complexeigenpair=libROM.NonSymmetricRightEigenSolve(m1)
    print("Complex Eigen pair ev",complexeigenpair.ev_real.get_data())
    print("Complex Eigen pair ev_imaginary",complexeigenpair.ev_imaginary.get_data())
    print("Complex Eigen pair eigs",complexeigenpair.eigs)
    np.testing.assert_allclose(complexeigenpair.ev_real.get_data(), [[-0.5932333119173846, 0.7864356987513791, -0.17202653679290827], [-0.679313061986337, -0.37436195478307094, 0.6311789687764832], [0.43198148275855325, 0.4912962635115682, 0.7563200248659911]])
    np.testing.assert_allclose(complexeigenpair.ev_imaginary.get_data(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(complexeigenpair.eigs, [(0.8548973087995788+0j), (2.4760236029181346+0j), (5.669079088282289+0j)])


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
    assert result.get_data() == [[450.0, 450.0], [450.0, 450.0]] 

def test_distributed():
    f = libROM.Matrix(2, 2, False)
    assert(not f.distributed())

    t = libROM.Matrix(2, 2, True)
    assert(t.distributed())

def test_balanced():
    two_by_two = libROM.Matrix(2, 2, False)
    assert(two_by_two.balanced())

def test_numColumns():
    two_by_two = libROM.Matrix(2, 2, False)
    two_by_three = libROM.Matrix(2, 3, False)
    three_by_two = libROM.Matrix(3, 2, False)

    assert(two_by_two.numColumns() == 2)
    assert(two_by_three.numColumns() == 3)
    assert(three_by_two.numColumns() == 2)

def test_setSize():
    one_by_one = libROM.Matrix(1, 1, False)

    assert(one_by_one.numRows() == 1)
    assert(one_by_one.numColumns() == 1)

    one_by_one.setSize(2, 2)
    assert(one_by_one.numRows() == 2)
    assert(one_by_one.numColumns() == 2)

def test_3arg_constructor_call():
    symmetric = np.array([[-2., 1.],
                          [1., -2.]])
    symmetric_matrix = libROM.Matrix(symmetric, False, True)
    assert(symmetric_matrix[0, 0] == -2.)
    assert(symmetric_matrix[0, 1] == 1.)
    assert(symmetric_matrix[1, 0] == 1.)
    assert(symmetric_matrix[1, 1] == -2.)

    asymmetric = np.array([[1., 1.],
                          [0., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    assert(asymmetric_matrix[0, 0] == 1.)
    assert(asymmetric_matrix[0, 1] == 1.)
    assert(asymmetric_matrix[1, 0] == 0.)
    assert(asymmetric_matrix[1, 1] == 1.)

def test_3arg_constructor_item():
    symmetric = np.array([[-2., 1.],
                          [1., -2.]])
    symmetric_matrix = libROM.Matrix(symmetric, False, True)
    assert(symmetric_matrix.item(0, 0) == -2.)
    assert(symmetric_matrix.item(0, 1) == 1.)
    assert(symmetric_matrix.item(1, 0) == 1.)
    assert(symmetric_matrix.item(1, 1) == -2.)

    asymmetric = np.array([[1., 1.],
                          [0., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    assert(asymmetric_matrix.item(0, 0) == 1.)
    assert(asymmetric_matrix.item(0, 1) == 1.)
    assert(asymmetric_matrix.item(1, 0) == 0.)
    assert(asymmetric_matrix.item(1, 1) == 1.)

def test_copy_constructor():
    asymmetric = np.array([[1., 1.],
                          [0., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    asymmetric_matrix2 = libROM.Matrix(asymmetric_matrix)

    assert(asymmetric_matrix2.numRows() == 2)
    assert(asymmetric_matrix2.numColumns() == 2)
    assert(asymmetric_matrix2.item(0, 0) == 1.)
    assert(asymmetric_matrix2.item(0, 1) == 1.)
    assert(asymmetric_matrix2.item(1, 0) == 0.)
    assert(asymmetric_matrix2.item(1, 1) == 1.)

def test_assignment():
    asymmetric = np.array([[1., 1.],
                          [0., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    asymmetric_matrix2 = libROM.Matrix()
    asymmetric_matrix2.__assign__(asymmetric_matrix)
    
    assert(asymmetric_matrix2.numRows() == 2)
    assert(asymmetric_matrix2.numColumns() == 2)
    assert(asymmetric_matrix2.item(0, 0) == 1.)
    assert(asymmetric_matrix2.item(0, 1) == 1.)
    assert(asymmetric_matrix2.item(1, 0) == 0.)
    assert(asymmetric_matrix2.item(1, 1) == 1.)
    assert(id(asymmetric_matrix2) is not id(asymmetric_matrix))

def test_get_first_n_columns():
    d_mat = np.array([[0.0, 1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0, 7.0],
                      [8.0, 9.0, 10.0, 11.0],
                      [12.0, 13.0, 14.0, 15.0]])
    matrix = libROM.Matrix(d_mat, False, True)
    truncated_matrix = matrix.getFirstNColumns(2)

    assert(truncated_matrix.numRows() == 4)
    assert(truncated_matrix.numColumns() == 2)
    assert(truncated_matrix.item(0, 0) == 0.)
    assert(truncated_matrix.item(0, 1) == 1.)
    assert(truncated_matrix.item(1, 0) == 4.)
    assert(truncated_matrix.item(1, 1) == 5.)
    assert(truncated_matrix.item(2, 0) == 8.)
    assert(truncated_matrix.item(2, 1) == 9.)
    assert(truncated_matrix.item(3, 0) == 12.)
    assert(truncated_matrix.item(3, 1) == 13.)

def test_pmatrix_mult():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    asymmetric_matrix2 = libROM.Matrix(asymmetric, False, True)
    result = asymmetric_matrix.mult(asymmetric_matrix2)

    assert(result.numRows() == 2)
    assert(result.numColumns() == 2)
    assert(result.item(0, 0) == 1.)
    assert(result.item(0, 1) == 0.)
    assert(result.item(1, 0) == 2.)
    assert(result.item(1, 1) == 1.)

def test_pmatrix_mult_output():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    asymmetric_matrix2 = libROM.Matrix(asymmetric, False, True)
    result = libROM.Matrix(2, 2, False)
    asymmetric_matrix.mult(asymmetric_matrix2, result)

    assert(result.numRows() == 2)
    assert(result.numColumns() == 2)
    assert(result.item(0, 0) == 1.)
    assert(result.item(0, 1) == 0.)
    assert(result.item(1, 0) == 2.)
    assert(result.item(1, 1) == 1.)

def test_pmatrix_transpose_mult_output():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)

    identity = np.array([[1., 0.],
                          [0., 1.]])
    identity_matrix = libROM.Matrix(identity, False, True)

    result = asymmetric_matrix.transposeMult(identity_matrix)
    assert(result.numRows() == 2)
    assert(result.numColumns() == 2)
    assert(result.item(0, 0) == 1.)
    assert(result.item(0, 1) == 1.)
    assert(result.item(1, 0) == 0.)
    assert(result.item(1, 1) == 1.)
    del result

    result = identity_matrix.transposeMult(asymmetric_matrix)
    assert(result.numRows() == 2)
    assert(result.numColumns() == 2)
    assert(result.item(0, 0) == 1.)
    assert(result.item(0, 1) == 0.)
    assert(result.item(1, 0) == 1.)
    assert(result.item(1, 1) == 1.)
    del result

    result = asymmetric_matrix.transposeMult(asymmetric_matrix)
    assert(result.numRows() == 2)
    assert(result.numColumns() == 2)
    assert(result.item(0, 0) == 2.)
    assert(result.item(0, 1) == 1.)
    assert(result.item(1, 0) == 1.)
    assert(result.item(1, 1) == 1.)
    del result

def test_void_transpose_mult_output():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)

    identity = np.array([[1., 0.],
                          [0., 1.]])
    identity_matrix = libROM.Matrix(identity, False, True)

    result = libROM.Matrix(2, 2, False)

    asymmetric_matrix.transposeMult(identity_matrix, result)
    assert(result.numRows() == 2)
    assert(result.numColumns() == 2)
    assert(result.item(0, 0) == 1.)
    assert(result.item(0, 1) == 1.)
    assert(result.item(1, 0) == 0.)
    assert(result.item(1, 1) == 1.)

    identity_matrix.transposeMult(asymmetric_matrix, result)
    assert(result.numRows() == 2)
    assert(result.numColumns() == 2)
    assert(result.item(0, 0) == 1.)
    assert(result.item(0, 1) == 0.)
    assert(result.item(1, 0) == 1.)
    assert(result.item(1, 1) == 1.)

    asymmetric_matrix.transposeMult(asymmetric_matrix, result)
    assert(result.numRows() == 2)
    assert(result.numColumns() == 2)
    assert(result.item(0, 0) == 2.)
    assert(result.item(0, 1) == 1.)
    assert(result.item(1, 0) == 1.)
    assert(result.item(1, 1) == 1.)

def test_void_inverse():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    asymmetric_matrix_inverse = libROM.Matrix(2, 2, False)
    asymmetric_matrix.inverse(asymmetric_matrix_inverse)

    assert(asymmetric_matrix_inverse.numRows() == 2)
    assert(asymmetric_matrix_inverse.numColumns() == 2)
    assert(asymmetric_matrix_inverse.item(0, 0) == 1.)
    assert(asymmetric_matrix_inverse.item(0, 1) == 0.)
    assert(asymmetric_matrix_inverse.item(1, 0) == -1.)
    assert(asymmetric_matrix_inverse.item(1, 1) == 1.)

def test_inverse_in_place():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    asymmetric_matrix.invert()

    assert(asymmetric_matrix.numRows() == 2)
    assert(asymmetric_matrix.numColumns() == 2)
    assert(asymmetric_matrix.item(0, 0) == 1.)
    assert(asymmetric_matrix.item(0, 1) == 0.)
    assert(asymmetric_matrix.item(1, 0) == -1.)
    assert(asymmetric_matrix.item(1, 1) == 1.)

def test_pmatrix_inverse():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    asymmetric_matrix_inverse = asymmetric_matrix.inverse()

    assert(asymmetric_matrix_inverse.numRows() == 2)
    assert(asymmetric_matrix_inverse.numColumns() == 2)
    assert(asymmetric_matrix_inverse.item(0, 0) == 1.)
    assert(asymmetric_matrix_inverse.item(0, 1) == 0.)
    assert(asymmetric_matrix_inverse.item(1, 0) == -1.)
    assert(asymmetric_matrix_inverse.item(1, 1) == 1.)

class SecondDifferenceMatrix(libROM.Matrix):
    def __init__(self, size):
        mat = np.diag(np.full(size, 2.)) + np.diag(-np.ones(size - 1), 1) + np.diag(-np.ones(size - 1), -1)
        super().__init__(mat, False, False)
        return
    
class PermutedIdentityMatrix(libROM.Matrix):
    def __init__(self, permutation, is_inverse=False):
        super().__init__(len(permutation), len(permutation), False)
        self.fill(0.)

        if (is_inverse):
            for k, perm in enumerate(permutation):
                self[perm, k] = 1
        else:
            for k, perm in enumerate(permutation):
                self[k, perm] = 1

def test_qrcp_pivots_transpose():
    from mpi4py import MPI

    size = 4
    permutation = np.array([0, 1, 2, 3])
    permuted_identity = PermutedIdentityMatrix(permutation, False)

    row_pivot = [0] * size
    row_pivot_owner = [0] * size
    row_pivots_requested = size
    row_pivot, row_pivot_owner = permuted_identity.qrcp_pivots_transpose(row_pivot, row_pivot_owner, row_pivots_requested)

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    assert(len(row_pivot) == size)
    for i in range(row_pivots_requested):
        assert row_pivot_owner[i] == my_rank
        assert row_pivot[i] < 5
    permutation=[0,1,2,3] 
    assert np.array_equal(row_pivot, permutation)

def test_mult_Vector():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    v_data = np.array([2., 4.])
    v = libROM.Vector(v_data, False, True)

    w = asymmetric_matrix.mult(v)

    assert(not w.distributed())
    assert(w.dim() == 2)
    assert(w[0] == 2.)
    assert(w[1] == 6.)

def test_mult_Vector_arg():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    v_data = np.array([2., 4.])
    v = libROM.Vector(v_data, False, True)
    w = libROM.Vector(2, False)

    asymmetric_matrix.mult(v, w)

    assert(not w.distributed())
    assert(w.dim() == 2)
    assert(w[0] == 2.)
    assert(w[1] == 6.)

def test_multPlus():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    v_data = np.array([2., 4.])
    v = libROM.Vector(v_data, False, True)
    w_data = np.array([0.5, 1.])
    w = libROM.Vector(w_data, False, True)

    asymmetric_matrix.multPlus(v, w, 2.0)

    assert(v[0] == 3.)
    assert(v[1] == 7.)

def test_transposeMult():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    v_data = np.array([2., 4.])
    v = libROM.Vector(v_data, False, True)

    w = asymmetric_matrix.transposeMult(v)

    assert(not w.distributed())
    assert(w.dim() == 2)
    assert(w[0] == 6.)
    assert(w[1] == 4.)

def test_transposeMult_arg():
    asymmetric = np.array([[1., 0.],
                          [1., 1.]])
    asymmetric_matrix = libROM.Matrix(asymmetric, False, True)
    v_data = np.array([2., 4.])
    v = libROM.Vector(v_data, False, True)
    w = libROM.Vector(2, False)

    asymmetric_matrix.transposeMult(v, w)

    assert(not w.distributed())
    assert(w.dim() == 2)
    assert(w[0] == 6.)
    assert(w[1] == 4.)

def test_outerProduct_serial():
    v_data = np.array([1., 2.])
    w_data = np.array([3., 4., 5.])
    v = libROM.Vector(v_data, False, True)
    w = libROM.Vector(w_data, False, True)

    vw = libROM.outerProduct(v, w)
    assert(vw.numRows() == 2)
    assert(vw.numColumns() == 3)
    assert(not vw.distributed())
    assert(np.array_equal(vw.getData(), [[3., 4., 5.],
                                         [6., 8., 10.]]))

def test_DiagonalMatrixFactory():
    w = libROM.Vector([1., 2., 3.], False, True)

    diagonalMatrix = libROM.DiagonalMatrixFactory(w)
    assert(diagonalMatrix.distributed() == w.distributed())
    assert(diagonalMatrix.numRows() == w.dim())
    assert(diagonalMatrix.numColumns() == w.dim())
    assert(np.array_equal(diagonalMatrix.getData(), [[1., 0., 0.],
                                                     [0., 2., 0.],
                                                     [0., 0., 3.]]))

def test_IdentityMatrixFactory():
    w = libROM.Vector([1., 2., 3.], False, True)

    identityMatrix = libROM.IdentityMatrixFactory(w)
    assert(identityMatrix.distributed() == w.distributed())
    assert(identityMatrix.numRows() == w.dim())
    assert(identityMatrix.numColumns() == w.dim())
    assert(np.array_equal(identityMatrix.getData(), [[1., 0., 0.],
                                                     [0., 1., 0.],
                                                     [0., 0., 1.]]))
    
def test_distribute_and_gather():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    num_procs = comm.Get_size()

    total_rows = 5
    answer = libROM.Matrix(total_rows, total_rows, False)
    assert(not answer.distributed())
    for i in range(total_rows):
        for j in range(total_rows):
            answer[i, j] = float(i * j)
    local_rows = utils.split_dimension(total_rows, comm)
    total_rows_check, row_offsets = utils.get_global_offsets(local_rows, comm)
    assert(total_rows == total_rows_check)

    test = libROM.Matrix(answer)
    test.distribute(local_rows)
    assert(test.distributed())
    for local_i in range(local_rows):
        global_i = row_offsets[my_rank] + local_i
        for j in range(answer.numColumns()):
            assert(test[local_i, j] == answer[global_i, j])

    test.gather()
    assert(not test.distributed())
    assert(np.array_equal(test.getData(), answer.getData()))


def test_matrix_orthogonalize():
    # Matrix data to orthonormalize
    d_mat = np.array([[3.5, 7.1, 0.0, 0.0],
                      [0.0, 1.9, 8.3, 0.0],
                      [0.0, 0.0, 5.7, 4.6],
                      [0.0, 0.0, 0.0, 3.2]])

    # target matrix data
    d_mat2 = np.eye(4)

    matrix = libROM.Matrix(d_mat, False, False)
    target = libROM.Matrix(d_mat2, False, False)

    matrix.orthogonalize()

    assert(np.allclose(matrix.getData(), target))


def test_matrix_orthogonalize4():
    # Matrix data to orthonormalize
    d_mat = np.array([[3.5, 7.1, 0.0, 0.0],
                      [0.0, 1.9, 8.3, 1.0e-14],
                      [0.0, 0.0, 5.7, 1.0+1.0e-14],
                      [0.0, 0.0, 0.0, 0.0]])

    # target matrix data
    d_mat2 = np.eye(4)
    d_mat2[3][3] = 0.0

    matrix = libROM.Matrix(d_mat, False, False)
    target = libROM.Matrix(d_mat2, False, False)

    matrix.orthogonalize(True)

    assert(np.allclose(matrix.getData(), target))


def test_matrix_orthogonalize_last():
    # Matrix data to orthonormalize
    d_mat = np.array([[1.0, 0.0, 0.0, 1.3],
                      [0.0, 1.0, 0.0, 4.7],
                      [0.0, 0.0, 1.0, 2.5],
                      [0.0, 0.0, 0.0, 7.3]])

    # target matrix data
    d_mat2 = np.eye(4)

    matrix = libROM.Matrix(d_mat, False, False)
    target = libROM.Matrix(d_mat2, False, False)

    matrix.orthogonalize_last()

    assert(np.allclose(matrix.getData(), target))


def test_matrix_orthogonalize_last2():
    # Matrix data to orthonormalize
    d_mat = np.array([[1.0, 0.0, 0.0, 1.3],
                      [0.0, 1.0, 0.0, 4.7],
                      [0.0, 0.0, 1.0, 2.5],
                      [0.0, 0.0, 0.0, 7.3]])

    # target matrix data
    d_mat2 = np.eye(4)

    matrix = libROM.Matrix(d_mat, False, False)
    target = libROM.Matrix(d_mat2, False, False)

    matrix.orthogonalize_last(-1, True)

    assert(np.allclose(matrix.getData(), target))


def test_matrix_orthogonalize_last4():
    # Matrix data to orthonormalize
    d_mat = np.array([[1.0, 0.0, 0.0, 1.3],
                      [0.0, 1.0, 0.0, 4.7],
                      [0.0, 0.0, 9.8, 2.5],
                      [0.0, 0.0, 0.0, 7.3]])

    # target matrix data
    d_mat2 = np.eye(4)

    matrix = libROM.Matrix(d_mat, False, False)
    target = libROM.Matrix(d_mat2, False, False)

    matrix.orthogonalize_last(3, True)
    matrix.orthogonalize_last(4, True)

    assert(np.allclose(matrix.getData(), target))


if __name__ == '__main__':
    pytest.main()