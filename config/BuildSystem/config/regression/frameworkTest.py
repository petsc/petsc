#!/usr/bin/env python
import nargs

import os
import unittest

nargs.setInteractive(0)

class FrameworkTest (unittest.TestCase):
  '''Tests basic capabilities of a Conf>igure Framework'''
  def __init__(self, methodName = 'runTest'):
    for f in ['RDict.db', 'RDict.log']:
      if os.path.exists(f):
        os.remove(f)
    unittest.TestCase.__init__(self, methodName)
    return

  def setUp(self):
    import config.framework

    self.framework = config.framework.Framework()
    self.framework.argDB['noOutput'] = 1
    return

  def tearDown(self):
    import logger

    self.framework.argDB.clear()
    files = [self.framework.header]
    if not self.framework.logName is None:
      files.extend([self.framework.logName, self.framework.logName+'.bkp'])
    for f in files:
      if os.path.exists(f):
        os.remove(f)
    if not logger.Logger.defaultLog is None:
      logger.Logger.defaultLog.close()
      logger.Logger.defaultLog = None
    self.framework = None
    return

  def testEmptyConfigure(self):
    '''Verify that an empty configure completes successfully'''
    self.assert_(self.framework.configure())
    return

  def testLogName(self):
    '''Verify that the log file is created with the correct name'''
    self.framework.logName = 'matt.log'
    self.failUnless(self.framework.configure())
    self.failUnless(os.path.isfile(self.framework.logName))
    return

  def testFullDefaultConfigure(self):
    '''Verify that a configure with all the default modules works correctly'''
    import config.base

    mod = config.base.Configure(self.framework)
    self.framework.addChild(mod)
    for package in os.listdir(os.path.dirname(config.__file__)):
      (packageName, ext) = os.path.splitext(package)
      if not packageName.startswith('.') and not packageName.startswith('#') and not packageName.endswith('-old') and ext == '.py' and not packageName in ['__init__', 'base', 'framework']:
        packageObj = self.framework.require('config.'+packageName, mod)
    self.failUnless(self.framework.configure())
    return

if __name__ == '__main__':
  unittest.main()
