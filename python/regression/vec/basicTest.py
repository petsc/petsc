#!/usr/bin/env python
import user
import importer

import unittest
import Numeric

class BasicTest(unittest.TestCase):
  '''Tests the Vec implementation'''
  def __init__(self, methodName):
    unittest.TestCase.__init__(self, methodName)
    import PETSc.Base
    import atexit

    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def setUp(self):
    import PETSc.Vec
    self.vec = PETSc.Vec.Vec()
    self.vec2 = PETSc.Vec.Vec()
    return

  def tearDown(self):
    self.vec = None
    self.vec2 = None
    return

  def testDot(self):
    '''Verify that the dot product works correctly'''
    self.vec.setSizes(-1, 14)
    self.vec.setUp()
    self.vec.set(2)
    self.vec2.setSizes(-1, 14)
    self.vec2.setUp()
    self.vec2.set(7)
    self.assertEquals(self.vec.dot(self.vec2), 14*14)
    return

  def testNorm(self):
    '''Verify that the norm works correctly'''
    from PETSc.NormType import NORM_2
    self.vec.setSizes(-1, 16)
    self.vec.setUp()
    self.vec.set(2)
    self.assertEquals(self.vec.norm(NORM_2), 8)
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
