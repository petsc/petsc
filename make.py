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
    return

  def defineDirectories(self):
    self.directories['main'] = self.getRoot()
    self.directories['sidl'] = os.path.join(self.directories['main'], 'sidl')
    return

  def defineFileSets(self):
    self.filesets['sidl'] = fileset.ExtensionFileSet(self.directories['sidl'], '.sidl')
    return

  def setupBuild(self):
    self.defineDirectories()
    self.defineFileSets()

    sidl = self.getSIDLDefaults()
    sidl.addServerLanguage('C++')
    sidl.addClientLanguage('C++')
    sidl.addClientLanguage('Python')
    return

if __name__ ==  '__main__':
  pm = PetscMake(sys.argv[1:])
  pm.main()
  pm.install()
