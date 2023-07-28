import pytest
import numpy
# import sys
# import os.path as pth
# sys.path.append(pth.join(pth.dirname(pth.abspath(__file__)), "../")) #sys.path.append("..")
import sys
sys.path.append("../build")
import pylibROM.linalg as libROM

# Create two Vector objects
v1 = libROM.Vector(3, False) 
v2 = libROM.Vector(3, False)

# Set the values for v1 and v2
v1.fill(1.0)
v2.fill(2.0)

# Print the initial data for v1 and v2
print("Initial data for v1:", v1.getData())
print("Initial data for v2:", v2.getData())

# Use the addition operator
v1 += v2

# Print the updated data for v1
print("Updated data for v1 after addition:", v1.getData())

#set every element to a scalar
v1.__set_scalar__(2)

# Print the updated data for v1
print("Updated data for v1 after setting every element to a scalar:", v1.getData())

#scaling every element by a scalar
v1.__scale__(3)

# Print the updated data for v1
print("Updated data for v1 after scaling every element by a scalar:", v1.getData())

#set size 
v1.setSize(5)
v1.__set_scalar__(2)

# Print the updated data for v1
print("Updated data for v1 after setting the size:", v1.getData())

#tranformers
# Define the transformer function
def transformer(size, vector):
    for i in range(size):
        vector[i] = vector[i] ** 2 
v1.transform(transformer)
print("Updated data for v1 after tranforming:", v1.getData())


print("distributed:", v1.distributed())

print("v1 dim:", v1.dim())

#inner product
v2.setSize(5)
v2.__set_scalar__(3)
print("inner product of v1 &v2", v1.inner_product(v2))

# Call the norm function
norm = v1.norm()
print("Norm:", norm)

# Call the norm2 function
norm2 = v1.norm2()
print("Norm squared:", norm2)

# Call the normalize function
v1.normalize()
normalized_data = v1.getData()
print("Normalized data:", normalized_data)

#add two vectors
#Test the first version of plus
result=v1.plus(v2)
print("Result vector after addition of v1 and v2 ", result.getData())

# Test the third version of plus
v1.plus(v2,v1)
print("Updated data for v1 after addition of v1 and v2 ", v1.getData())

#Test the first version of plusAx
result1 = v1.plusAx(2.0, v2)
print("Result vector", result1.getData())


#Test the first version of plusEqAx
v1.plusEqAx(2.0, v2)
print("Updated data for v1 after plusEqAx", v1.getData())

#Test the first version of minus
result=v1.minus(v2)
print("Result vector after subtraction of v1 and v2 ", result.getData())

# Test the third version of minus
v1.minus(v2,v1)
print("Updated data for v1 after subtraction of v1 and v2 ", v1.getData())


# Test the 'localMin' function
local_min = v1.localMin(0)

# Print the local minimum value
print("Local Minimum of v1:", local_min)


def test_plus():
    v1 = libROM.Vector(3, False)
    v2 = libROM.Vector(3, False)

    # Set the values for v1 and v2
    v1.fill(1.0)
    v2.fill(2.0)

    v1 += v2
    result = v1.getData()
    print(result)
    assert result == [3.0, 3.0, 3.0]

if __name__ == '__main__':
    pytest.main()




