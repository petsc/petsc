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
    sys.argv.append('-start_in_debugger')
    sys.argv.append('-mat_view')
    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def setUp(self):
    import PETSc.Mat
    self.mat = PETSc.Mat.Mat()
    return

  def tearDown(self):
    self.mat = None
    return

  def testMult(self):
    '''Verify that matrix-vector multiplication works correctly'''
    from PETSc.MatAssemblyType import MAT_FINAL_ASSEMBLY
    from PETSc.InsertMode import INSERT_VALUES
    import PETSc.Vec
    x = PETSc.Vec.Vec()
    x.setSizes(-1, 10)
    x.setUp()
    x.set(1)
    y = PETSc.Vec.Vec()
    y.setSizes(-1, 10)
    y.setUp()
    self.mat.setSizes(-1, -1, 10, 10)
    self.mat.setUp()
    for i in range(10):
      if i == 0:
        self.mat.setValues([i], [i, i+1], [2.0, -1.0], INSERT_VALUES)
      elif i == 9:
        self.mat.setValues([i], [i-1, i], [-1.0, 2.0], INSERT_VALUES)
      else:
        self.mat.setValues([i], [i-1, i, i+1], [-1.0, 2.0, -1.0], INSERT_VALUES)
    self.mat.assemblyBegin(MAT_FINAL_ASSEMBLY)
    self.mat.assemblyEnd(MAT_FINAL_ASSEMBLY)
    print 'Built matrix'
    self.mat.mult(x, y)
    print 'Did matvec'
    self.assertEquals(y.getArray(), [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0])
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
