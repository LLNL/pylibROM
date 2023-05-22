from pylibROM.linalg import Vector


# Create two Vector objects
v1 = Vector(3, False)
v2 = Vector(3, False)

# Set the values for v1 and v2
v1.fill(1.0)
v2.fill(2.0)

# Print the initial data for v1 and v2
print("Initial data for v1:", v1.get_data())
print("Initial data for v2:", v2.get_data())

# Use the addition operator
v1 += v2

# Print the updated data for v1
print("Updated data for v1 after addition:", v1.get_data())
