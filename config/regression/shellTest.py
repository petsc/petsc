#!/usr/bin/env python
import user

import unittest

class ShellTest (unittest.TestCase):
  '''Tests basic capabilities of Configure'''
  def setUp(self):
    return

  def tearDown(self):
    return

  def testTimeout(self):
    '''Verify that the timeout of a shell command is handled correctly'''
    import config.base

    timeout = 3
    self.assertRaises(RuntimeError, config.base.Configure.executeShellCommand, 'sleep '+str(timeout), timeout = timeout-1)
    self.assertEquals(0, config.base.Configure.executeShellCommand('sleep '+str(timeout), timeout = timeout+1)[2])
    return

if __name__ == '__main__':
  unittest.main()
