#!/usr/bin/env python
import bs
import BSTemplates.sidl
import fileset

import os
import sys
import argtest
import commands
import distutils.sysconfig
import string

def getpythoninclude():
    return distutils.sysconfig.get_python_inc()

def getpythonlib():
    lib = distutils.sysconfig.get_config_var('LIBPL')+"/"+distutils.sysconfig.get_config_var('LDLIBRARY')
    lib = string.split(lib,'.so')[0]+'.so'
    return lib

class PetscMake(bs.BS):
  def __init__(self, args = None):
    bs.BS.__init__(self, args)
    bs.argDB['PYTHON_INCLUDE'] = getpythoninclude()
    bs.argDB['PYTHON_LIB'] = getpythonlib()
    self.defineHelp()
    self.defineDirectories()
    self.defineFileSets()
    self.defineTargets()

  def install(self):
    # this is pitiful
    try:
      os.makedirs(bs.argDB['installlib'])
    except:
      pass
    (status, output) = commands.getstatusoutput('cp -f *.py '+bs.argDB['installlib'])
    (status, output) = commands.getstatusoutput('cp -rf BSTemplates '+bs.argDB['installlib'])

    try:
      os.makedirs(bs.argDB['installlib'])
    except:
      pass
    (status, output) = commands.getstatusoutput('cp -f lib/*.so '+bs.argDB['installlib'])
    
  def defineHelp(self):
    bs.argDB.setHelp('PYTHON_INCLUDE', 'The directory in which the Python headers were installed (like Python.h)')
    bs.argDB.setTester('PYTHON_INCLUDE',argtest.DirectoryTester())

    bs.argDB.setHelp('SIDLRUNTIME_DIR', 'The directory in which the SIDL runtime was installed')
    bs.argDB.setTester('SIDLRUNTIME_DIR',argtest.DirectoryNotNoneTester())

  def defineDirectories(self):
    self.directories['sidl'] = os.path.abspath('sidl')

  def defineFileSets(self):
    self.filesets['sidl'] = fileset.ExtensionFileSet(self.directories['sidl'], '.sidl')

  def defineTargets(self):
    babelDefaults = BSTemplates.sidl.CompileDefaults('bs', self.filesets['sidl'])
    babelDefaults.addServerLanguage('C++')
    babelDefaults.addClientLanguage('C++')
    babelDefaults.addClientLanguage('Python')

    self.targets['sidl']    = babelDefaults.getSIDLTarget()
    self.targets['compile'] = babelDefaults.getCompileTarget()
    self.targets['default'] = self.targets['compile']

if __name__ ==  '__main__':
  pm = PetscMake(sys.argv[1:])
  pm.main()
  pm.install()
