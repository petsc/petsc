#!/usr/bin/env python
import user
import importer

import unittest
import Numeric

class BasicTest(unittest.TestCase):
  '''Tests Base implementation'''
  def __init__(self, methodName):
    unittest.TestCase.__init__(self, methodName)
    import PETSc.Base
    import atexit

    PETSc.Base.Base.Initialize()
    atexit.register(PETSc.Base.Base.Finalize)
    return

  def setUp(self):
    import PETSc.PetscMap
    self.map = PETSc.PetscMap.PetscMap()
    return

  def tearDown(self):
    self.map = None
    return

  def testLocalSize(self):
    '''Verify that we can change the local size'''
    self.assertEquals(self.map.getLocalSize(), -1)
    self.map.setLocalSize(1001)
    self.assertEquals(self.map.getLocalSize(), 1001)
    self.map.setLocalSize(10)
    self.assertEquals(self.map.getLocalSize(), 10)
    return

  def testGlobalSize(self):
    '''Verify that we can change the global size'''
    self.assertEquals(self.map.getSize(), -1)
    self.map.setSize(1001)
    self.assertEquals(self.map.getSize(), 1001)
    self.map.setSize(10)
    self.assertEquals(self.map.getSize(), 10)
    return

  def testFullySpecified(self):
    '''Verify that fully specified maps work'''
    self.map.setLocalSize(10)
    self.map.setSize(10)
    self.map.setType('mpi')
    self.assertEquals(self.map.getLocalSize(), 10)
    self.assertEquals(self.map.getSize(), 10)
    return

  def testGloballySpecified(self):
    '''Verify that globally specified maps work'''
    self.map.setSize(10)
    self.map.setType('mpi')
    self.assertEquals(self.map.getLocalSize(), 10)
    self.assertEquals(self.map.getSize(), 10)
    return

  def testLocallySpecified(self):
    '''Verify that locally specified maps work'''
    self.map.setLocalSize(10)
    self.map.setType('mpi')
    self.assertEquals(self.map.getLocalSize(), 10)
    self.assertEquals(self.map.getSize(), 10)
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
