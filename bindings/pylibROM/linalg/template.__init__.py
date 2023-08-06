# If a pure python routine is added to pylibROM.linalg module,
# rename this file to __init__.py,
# and either define/import the python routine in this file.
# This will combine both c++ bindings/pure python routines into pylibROM.linalg module.

# For other c++ binding modules, change the module name accordingly.
from _pylibROM.linalg import *