import unittest
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
from petsc4py import PETSc

# --------------------------------------------------------------------


class TestStdout(unittest.TestCase):
    def testStdoutRedirect(self):
        for cycle in range(2):
            with self.subTest(cycle=cycle):
                self._checkStdoutRedirect()

    def _checkStdoutRedirect(self):
        newstdout = StringIO()
        newstderr = StringIO()
        a = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
        a_vec = PETSc.Vec().createWithArray(a, comm=PETSc.COMM_SELF)
        try:
            with redirect_stdout(newstdout), redirect_stderr(newstderr):
                PETSc._push_python_vfprintf()
                try:
                    a_vec.view()
                    v = PETSc.Viewer.STDERR(PETSc.COMM_SELF)
                    v.printfASCII('Error message')
                finally:
                    PETSc._pop_python_vfprintf()
        finally:
            a_vec.destroy()
        output = newstdout.getvalue()
        error = newstderr.getvalue()
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
