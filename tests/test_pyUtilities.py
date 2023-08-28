import sys
import pytest
sys.path.append("../build")
import numpy as np 
import _pylibROM.hyperreduction as hyperreduction

def test_Rowinfo():
    row_info = hyperreduction.RowInfo()
    row_info.row_val = 3
    row_info.row = 42
    row_info.proc = 2

    assert row_info.row_val == 3
    assert row_info.row == 42
    assert row_info.proc == 2

if __name__ == '__main__':
    pytest.main()