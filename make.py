#!/usr/bin/env python
import bs
import fileset

import os
import pwd
import sys
import commands
import string
import nargs

class PetscMake(bs.BS):
  def __init__(self, args = None):
    bs.BS.__init__(self, bs.Project('bs', 'bk://sidl.bkbits.net/BuildSystem', self.getRoot()), args)
    self.defineDirectories()
    self.defineFileSets()
    self.defineBuild()
    return

  def install(self):
    if not bs.argDB.has_key('install'): return
    bs.argDB.setType('installlib',nargs.ArgDir(0,'Location to install libraries'))
    bs.argDB.setType('installh',nargs.ArgDir(0,'Location to install include files'))
    bs.argDB.setType('installexamples',nargs.ArgDir(0,'Location to install examples'))
    try:
      os.makedirs(bs.argDB['installlib'])
      os.makedirs(bs.argDB['installh'])
      os.makedirs(bs.argDB['installexamples'])
    except:
      pass
    (status, output) = commands.getstatusoutput('cp -f *.py '+bs.argDB['installlib'])
    (status, output) = commands.getstatusoutput('cp -f lib/*.so '+bs.argDB['installlib'])
    return

  def defineDirectories(self):
    self.directories['main'] = self.getRoot()
    self.directories['sidl'] = os.path.join(self.directories['main'], 'sidl')
    return

  def defineFileSets(self):
    self.filesets['sidl'] = fileset.ExtensionFileSet(self.directories['sidl'], '.sidl')
    return

  def defineBuild(self):
    sidl = self.getSIDLDefaults()
    sidl.addServerLanguage('C++')
    sidl.addClientLanguage('C++')
    sidl.addClientLanguage('Python')
    return

if __name__ ==  '__main__':
  pm = PetscMake(sys.argv[1:])
  pm.main()
  pm.install()
