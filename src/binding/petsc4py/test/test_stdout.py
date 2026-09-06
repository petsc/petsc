import unittest
from io import StringIO
import sys
import numpy as np
from petsc4py import PETSc

# --------------------------------------------------------------------


class TestStdout(unittest.TestCase):
    def testStdoutRedirect(self):
        prevstdout = sys.stdout
        prevstderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        if __name__ != '__main__':
            PETSc._push_python_vfprintf()

        a = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
        a_vec = PETSc.Vec().createWithArray(a, comm=PETSc.COMM_SELF)
        a_vec.view()
        v = PETSc.Viewer.STDERR(PETSc.COMM_SELF)
        v.printfASCII('Error message')

        newstdout = sys.stdout
        newstderr = sys.stderr
        sys.stdout = prevstdout
        sys.stderr = prevstderr

        output = newstdout.getvalue()
        error = newstderr.getvalue()
        if __name__ != '__main__':
            PETSc._pop_python_vfprintf()
        stdoutshouldbe = """Vec Object: 1 MPI process
  type: seq
0.
0.
0.
"""
        stderrshouldbe = 'Error message'
        if PETSc._stdout_is_stderr():
            stdoutshouldbe = stdoutshouldbe + stderrshouldbe
            stderrshouldbe = ''
        self.assertEqual(output, stdoutshouldbe)
        self.assertEqual(error, stderrshouldbe)


# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
