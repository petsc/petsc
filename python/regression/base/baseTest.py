#!/usr/bin/env python
import user
import importer

import unittest
import Numeric

class BaseTest(unittest.TestCase):
  '''Tests Base implementation'''
  def __init__(self, methodName):
    unittest.TestCase.__init__(self, methodName)
    import PETSc.Base
    import atexit

    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def setUp(self):
    import PETSc.Base
    self.base = PETSc.Base.Base()
    return

  def tearDown(self):
    self.base = None
    return

  def testName(self):
    '''Verify that object naming works'''
    name = 'Matt'
    self.base.setName(name)
    self.assertEquals(self.base.getName(), name)
    return

  def testOptionsPrefix(self):
    '''Verify that we can manipulate the options prefix'''
    self.assertEquals(self.base.getOptionsPrefix(), None)
    self.base.setOptionsPrefix('matt_')
    self.assertEquals(self.base.getOptionsPrefix(), 'matt_')
    self.base.appendOptionsPrefix('dr_')
    self.assertEquals(self.base.getOptionsPrefix(), 'matt_dr_')
    self.base.setOptionsPrefix('')
    self.assertEquals(self.base.getOptionsPrefix(), '')
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
