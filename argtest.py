#!/usr/bin/env python


import atexit
import cPickle
import os
import sys
#
#   Provides classes to be used by argDB.setTester('argname',Tester()) to test for valid
# arguments.

class IntTester:
  def test(self,value):
      if not value:
	  return (1,'0')
      try:
	  int(value)
	  return (1,value)
      except ValueError:
	  return (0,None)

class DirectoryTester:
  def test(self,value):
      if not value:
	  return (1,None)
      if os.access(value,os.R_OK):
	  return (1,value)
      else:
	  print 'No directory named'+value
	  return (0,None)

class DirectoryNotNoneTester:
  def test(self,value):
      if not value:
	  print 'You must enter valid directory'
	  return (0,None)
      if os.access(value,os.R_OK):
	  return (1,value)
      else:
	  print 'No directory named'+value
	  return (0,None)
