#!/usr/bin/env python
import user
import importer

import unittest
import Numeric

class InitializationTest(unittest.TestCase):
  '''Tests static Base methods'''
  def __init__(self, methodName):
    unittest.TestCase.__init__(self, methodName)
    return

  def setUp(self):
    return

  def tearDown(self):
    return

  def testInitialize(self):
    '''Verify that PETSc initialization is working'''
    import PETSc.Base
    self.failIf(PETSc.Base.Base.Initialized())
    PETSc.Base.Base.Initialize()
    self.failUnless(PETSc.Base.Base.Initialized())
    self.failIf(PETSc.Base.Base.Finalized())
    PETSc.Base.Base.Finalize()
    self.failUnless(PETSc.Base.Base.Finalized())
    return

if __name__ == '__main__':
  import os
  import sys

  sys.path.append(os.path.abspath(os.getcwd()))
  if len(sys.argv) > 1:
    if not sys.argv[1] in globals():
      raise RuntimeError('No test class '+sys.argv[1]+' found.')
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(globals()[sys.argv[1]]))
    unittest.TextTestRunner(verbosity = 2).run(suite)
  else:
    unittest.main()
