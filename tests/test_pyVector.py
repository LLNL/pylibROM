import pytest
import numpy as np
import sys
try:
    # import pip-installed package
    import pylibROM
    import pylibROM.linalg as libROM 
except ModuleNotFoundError:
    # If pip-installed package is not found, import cmake-built package
    sys.path.append("../build")
    import _pylibROM as pylibROM
    import _pylibROM.linalg as libROM

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

def test_distributed():
    v = libROM.Vector(2, False)
    assert(not v.distributed())
    w = libROM.Vector(2, True)
    assert(w.distributed())

def test_dim():
    v = libROM.Vector(2, False)
    assert(v.dim() == 2)

def test_setSize():
    v = libROM.Vector(2, False)
    assert(v.dim() == 2)
    v.setSize(3)
    assert(v.dim() == 3)

def test_call_operator():
    v_data = np.array([1., 2.])
    v = libROM.Vector(v_data, False, True)
    assert(v[0] == 1.)
    assert(v[1] == 2.)

def test_item():
    v_data = np.array([1., 2.])
    v = libROM.Vector(v_data, False, True)
    assert(v.item(0) == 1.)
    assert(v.item(1) == 2.)

def test_copy_constructor():
    v_data = np.array([1., 2.])
    v = libROM.Vector(v_data, False, True)
    w = libROM.Vector(v)

    assert(not w.distributed())
    assert(w.dim() == 2)
    assert(w[0] == 1.)
    assert(w[1] == 2.)

def test_copy_assignment_operator():
    v_data = np.array([1., 2.])
    v = libROM.Vector(v_data, False, True)
    w = v

    assert(not w.distributed())
    assert(w.dim() == 2)
    assert(w[0] == 1.)
    assert(w[1] == 2.)

def test_norm():
    v = libROM.Vector(2, False)

    v[0] = 1.0
    v[1] = 1.0
    assert(abs(v.norm() - np.sqrt(2.)) <= 1.0e-15)

    v[0] = -1.0
    v[1] = 1.0
    assert(abs(v.norm() - np.sqrt(2.)) <= 1.0e-15)

    v[0] = -3.0
    v[1] = 4.0
    assert(abs(v.norm() - 5.) <= 1.0e-15)

    v[0] = 5.0
    v[1] = -12.0
    assert(abs(v.norm() - 13.) <= 1.0e-15)

def test_normalize():
    v = libROM.Vector(2, False)
    v[0] = 3.0
    v[1] = 4.0

    assert(abs(v.normalize() - 5.) <= 1.0e-15)
    assert(v[0] == 0.6)
    assert(v[1] == 0.8)
    assert(v.norm() == 1.)

def test_inner_product():
    v = libROM.Vector(2, False)
    v[0] = 1.0
    v[1] = 1.0
    w = libROM.Vector(2, False)
    w[0] = -1.0
    w[1] = 1.0
    x = libROM.Vector(2, False)
    x[0] = 3.0
    x[1] = 4.0
    y = libROM.Vector(2, False)
    y[0] = 5.0
    y[1] = 12.0

    assert(v.inner_product(v) ==  2.)
    assert(v.inner_product(w) ==  0.)
    assert(v.inner_product(x) ==  7.)
    assert(v.inner_product(y) == 17.)
    assert(w.inner_product(v) ==  0.)
    assert(w.inner_product(w) ==  2.)
    assert(w.inner_product(x) ==  1.)
    assert(w.inner_product(y) ==  7.)
    assert(x.inner_product(v) ==  7.)
    assert(x.inner_product(w) ==  1.)
    assert(x.inner_product(x) == 25.)
    assert(x.inner_product(y) == 63.)
    assert(y.inner_product(v) == 17.)
    assert(y.inner_product(w) ==  7.)
    assert(y.inner_product(x) == 63.)
    assert(y.inner_product(y) ==169.)

def test_plus():
    v = libROM.Vector(2, False)
    v[0] = 1.0
    v[1] = 1.0
    w = libROM.Vector(2, False)
    w[0] = -1.0
    w[1] = 1.0
    x = libROM.Vector(2, False)
    x[0] = 3.0
    x[1] = 4.0
    y = libROM.Vector(2, False)
    y[0] = 5.0
    y[1] = 12.0

    result = v.plus(v)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 2.)
    assert(result[1] == 2.)
    del result

    result = v.plus(w)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 0.)
    assert(result[1] == 2.)
    del result

    result = v.plus(x)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 4.)
    assert(result[1] == 5.)
    del result

    result = v.plus(y)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 6.)
    assert(result[1] == 13.)
    del result

def test_plusAx():
    v = libROM.Vector(2, False)
    v[0] = 1.0
    v[1] = 1.0
    w = libROM.Vector(2, False)
    w[0] = -1.0
    w[1] = 1.0
    x = libROM.Vector(2, False)
    x[0] = 3.0
    x[1] = 4.0
    y = libROM.Vector(2, False)
    y[0] = 5.0
    y[1] = 12.0

    result = v.plusAx(1.0, v)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 2.)
    assert(result[1] == 2.)
    del result

    result = v.plusAx(1.0, w)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 0.)
    assert(result[1] == 2.)
    del result

    result = v.plusAx(1.0, x)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 4.)
    assert(result[1] == 5.)
    del result

    result = v.plusAx(1.0, y)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 6.)
    assert(result[1] == 13.)
    del result

def test_pluxEqAx():
    v = libROM.Vector(2, False)
    v[0] = 1.0
    v[1] = 1.0
    w = libROM.Vector(2, False)
    w[0] = -1.0
    w[1] = 1.0
    x = libROM.Vector(2, False)
    x[0] = 3.0
    x[1] = 4.0
    y = libROM.Vector(2, False)
    y[0] = 5.0
    y[1] = 12.0

    v.plusEqAx(1.0, v)
    assert(not v.distributed())
    assert(v.dim() == 2)
    assert(v[0] == 2.)
    assert(v[1] == 2.)

    v[0] = 1.0
    v[1] = 1.0
    v.plusEqAx(1.0, w)
    assert(not v.distributed())
    assert(v.dim() == 2)
    assert(v[0] == 0.)
    assert(v[1] == 2.)

    v[0] = 1.0
    v[1] = 1.0
    v.plusEqAx(1.0, x)
    assert(not v.distributed())
    assert(v.dim() == 2)
    assert(v[0] == 4.)
    assert(v[1] == 5.)

    v[0] = 1.0
    v[1] = 1.0
    v.plusEqAx(1.0, y)
    assert(not v.distributed())
    assert(v.dim() == 2)
    assert(v[0] == 6.)
    assert(v[1] == 13.)

def test_minus():
    v = libROM.Vector(2, False)
    v[0] = 1.0
    v[1] = 1.0
    w = libROM.Vector(2, False)
    w[0] = -1.0
    w[1] = 1.0
    x = libROM.Vector(2, False)
    x[0] = 3.0
    x[1] = 4.0
    y = libROM.Vector(2, False)
    y[0] = 5.0
    y[1] = 12.0

    result = v.minus(v)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 0.)
    assert(result[1] == 0.)
    del result

    result = v.minus(w)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 2.)
    assert(result[1] == 0.)
    del result

    result = v.minus(x)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == -2.)
    assert(result[1] == -3.)
    del result

    result = v.minus(y)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == -4.)
    assert(result[1] == -11.)
    del result

def test_mult():
    v = libROM.Vector(2, False)
    v[0] = 1.0
    v[1] = 1.0
    w = libROM.Vector(2, False)
    w[0] = -1.0
    w[1] = 1.0
    x = libROM.Vector(2, False)
    x[0] = 3.0
    x[1] = 4.0
    y = libROM.Vector(2, False)
    y[0] = 5.0
    y[1] = 12.0

    result = v.mult(2.)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 2.)
    assert(result[1] == 2.)
    del result

    result = w.mult(-5.)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 5.)
    assert(result[1] == -5.)
    del result

    result = x.mult(3.)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 9.)
    assert(result[1] == 12.)
    del result

    result = y.mult(0.5)
    assert(not result.distributed())
    assert(result.dim() == 2)
    assert(result[0] == 2.5)
    assert(result[1] == 6.)
    del result

if __name__ == '__main__':
    pytest.main()
