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
print("Initial data for v1:", v1.get_data())
print("Initial data for v2:", v2.get_data())

# Test copy constructor
v3 = libROM.Vector(v1)  
print("Create a new Vector(v3) using the copy constructor: ",v3.get_data())  

# Test assignment operator
v3 = v2  
print("Assign v2 to v3 using the assignment operator: ",v3.get_data())  

# Use the addition operator
v1 += v2

# Print the updated data for v1
print("Updated data for v1 after addition:", v1.get_data()) 

# Subtract v2 from v1 using the -= operator
v1 -= v2
print("Updated data for v1 after subtraction:", v1.get_data())  

#set every element to a scalar
v1.__set_scalar__(2)

# Print the updated data for v1
print("Updated data for v1 after setting every element to a scalar:", v1.get_data())

#scaling every element by a scalar
v1.__scale__(3)

# Print the updated data for v1
print("Updated data for v1 after scaling every element by a scalar:", v1.get_data())

#set size 
v1.setSize(5)
v1.__set_scalar__(2)

# Print the updated data for v1
print("Updated data for v1 after setting the size:", v1.get_data())

#tranformers
# Define the transformer function
def transformer(size, vector):
    for i in range(size):
        vector[i] = vector[i] ** 2 
v1.transform(transformer)
print("Updated data for v1 after tranforming:", v1.get_data())


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
normalized_data = v1.get_data()
print("Normalized data:", normalized_data)

#add two vectors
#Test the first version of plus
result=v1.plus(v2)
print("Result vector after addition of v1 and v2 ", result.get_data())

# Test the third version of plus
v1.plus(v2,v1)
print("Updated data for v1 after addition of v1 and v2 ", v1.get_data())


# Test plusAx function that returns a new Vector
result1 = v1.plusAx(2.0, v2)
print("Result vector of plusAx function", result1.get_data())

# Test plusAx function that modifies an existing Vector
result2 = libROM.Vector()
v1.plusAx(2.0, v2, result2)
print("Result vector(existing vector) of plusAx function", result2.get_data())


#Test the first version of plusEqAx
v1.plusEqAx(2.0, v2)
print("Updated data for v1 after plusEqAx", v1.get_data())

#Test the first version of minus
result=v1.minus(v2)
print("Result vector after subtraction of v1 and v2 ", result.get_data())

# Test the third version of minus
v1.minus(v2,v1)
print("Updated data for v1 after subtraction of v1 and v2 ", v1.get_data())


# Test mult function that returns a new Vector
result1 = v1.mult(2.0)
print("The result vector of multiplication of vector v1 by a factor of 2.0",result1.get_data())

# Test mult function that modifies an existing Vector
result2 = libROM.Vector()
v1.mult(2.0, result2)
print("The result vector(existing vector) of multiplication of vector v1 by a factor of 2.0",result2.get_data())

v1=libROM.Vector(2,False)
v1.__setitem__(0, 0)
v1.__setitem__(1, 2.0)
print("Set Item(1) of vector v1 to 2.0 ",v1.get_data())
print("Get Item (1) of vector v1",v1.__getitem__(1) )
value= v1(1)
print("value",value)
print("call function",v1.__call__(1))

# Test the 'localMin' function
local_min = v1.localMin(0)

# Print the local minimum value
print("Local Minimum of v1:", local_min)


v2=libROM.Vector(2,False)
v2.fill(4.0)
pointers = [v1, v2]

# Test getCenterPoint with vector of Vector pointers
center_point_1 = libROM.getCenterPoint(pointers, True)
print("GetCenterPoint with vector of Vector pointers(v1,v2)",center_point_1)

# Create a vector of Vector objects
objects = [v1, v2]

# Test getCenterPoint with vector of Vector objects
center_point_2 = libROM.getCenterPoint(objects, True)
print("GetCenterPoint with vector of Vector objects",center_point_2)

# Create a test Vector object
test_point = libROM.Vector(2,False)
test_point.fill(3.0)

# Test getClosestPoint with vector of Vector pointers
closest_point_1 = libROM.getClosestPoint(pointers, test_point)
print("GetClosestPoint with vector of Vector pointers",closest_point_1)

# Test getClosestPoint with vector of Vector objects
closest_point_2 = libROM.getClosestPoint(objects, test_point)
print("GetClosestPoint with vector of Vector objects",closest_point_2)