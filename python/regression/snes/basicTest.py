#!/usr/bin/env python
import user
import importer

import unittest
import Numeric

class BasicTest(unittest.TestCase):
  '''Tests the Mat implementation'''
  def __init__(self, methodName):
    unittest.TestCase.__init__(self, methodName)
    import PETSc.Base
    import atexit

    import sys
    #sys.argv.append('-start_in_debugger')
    sys.argv.append('-snes_monitor')
    sys.argv.extend(['-pc_type', 'jacobi'])
    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def setUp(self):
    import PETSc.SNES
    self.snes = PETSc.SNES.SNES()
    return

  def tearDown(self):
    self.snes = None
    return

  def testSolve(self):
    '''Verify that we can solve a simple set of nonlinear equations
       / x^2 - y + z - 10 \      /3\                                        /5\
       |  3x - y^2 - z^3  |  x = |1|  which needs a starting guess of x_0 = |5|
       \   y^3 - 2z + 3   /      \2/                                        \5/
    '''
    import PETSc.Vec
    r = PETSc.Vec.Vec()
    r.setSizes(-1, 3)
    r.setUp()
    def func(snes, X, F):
      x = X.getArray()
      f = F.getArray()
      f[0] = x[0]*x[0] - x[1] + x[2] - 10.0
      f[1] = 3.0*x[0] - x[1]*x[1] - x[2]*x[2]*x[2]
      f[2] = x[1]*x[1]*x[1] - 2.0*x[2] + 3.0
      X.restoreArray(x)
      F.restoreArray(f)
      return 0
    self.snes.setFunction(r, func)
    import PETSc.Mat
    J = PETSc.Mat.Mat()
    J.setSizes(-1, -1, 3, 3)
    J.setUp()
    def jac(snes, X, J, M, structureFlag):
      from PETSc.MatAssemblyType import MAT_FINAL_ASSEMBLY
      from PETSc.InsertMode import INSERT_VALUES
      x = X.getArray()
      J.setValues([0, 1, 2], [0, 1, 2], [2.0*x[0], -1.0, 1.0, 3.0, -2.0*x[1], -3.0*x[2]*x[2], 0.0, 3.0*x[1]*x[1], -2.0], INSERT_VALUES)
      X.restoreArray(x)
      J.assemblyBegin(MAT_FINAL_ASSEMBLY)
      J.assemblyEnd(MAT_FINAL_ASSEMBLY)
      return 0
    self.snes.setJacobian(J, J, jac)
    self.snes.setFromOptions()
    x = PETSc.Vec.Vec()
    x.setSizes(-1, 3)
    x.setUp()
    x.set(5.0)
    self.snes.solve(None, x)
    [self.assertAlmostEqual(xhat_i, x_i, 6) for xhat_i, x_i in zip(x.getArray(), [3.0, 1.0, 2.0])]
    return

if __name__ == '__main__':
  import os
  import sys

  sys.path.append(os.path.abspath(os.getcwd()))
  if len(sys.argv) > 1 and not sys.argv[1][0] == '-':
    if not sys.argv[1] in globals():
      raise RuntimeError('No test class '+sys.argv[1]+' found.')
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(globals()[sys.argv[1]]))
    unittest.TextTestRunner(verbosity = 2).run(suite)
  else:
    unittest.main()
