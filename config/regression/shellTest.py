#!/usr/bin/env python
import user

import unittest

class ShellTest (unittest.TestCase):
  '''Tests basic capabilities of TOPS.Vector'''
  def setUp(self):
    import config.base
    import config.framework

    self.framework = config.framework.Framework(loadArgDB = 0)
    self.framework.setupArguments(self.framework.clArgs)
    self.framework.setupLogging()
    self.configure = config.base.Configure(self.framework)
    return

  def tearDown(self):
    del self.configure
    del self.framework
    return

  def testTimeout(self):
    '''Verify that the timeout of a shell command is handled correctly'''
    timeout = 3
    self.assertRaises(RuntimeError, self.configure.executeShellCommand, 'sleep '+str(timeout), timeout = timeout-1)
    self.assertEquals(0, self.configure.executeShellCommand('sleep '+str(timeout), timeout = timeout+1)[2])
    return

if __name__ == '__main__':
  unittest.main()
