#!/usr/bin/env python
import bs
import fileset
import BSTemplates.babelTargets
import BSTemplates.scandalTargets
import BSTemplates.compileTargets

import os
import os.path
import pwd
import sys
import commands
import string
import nargs

class PetscMake(bs.BS):
  def __init__(self, args = None):
    bs.BS.__init__(self, args)
    self.defineDirectories()
    self.defineFileSets()
    self.defineTargets()

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
    
  def defineDirectories(self):
    self.directories['sidl'] = os.path.abspath('sidl')

  def defineFileSets(self):
    self.filesets['sidl'] = fileset.ExtensionFileSet(self.directories['sidl'], '.sidl')

  def defineTargets(self):
    if bs.argDB.has_key('babelCrap') and bs.argDB['babelCrap']:
      sidl = BSTemplates.babelTargets.Defaults('bs', self.filesets['sidl'])
    else:
      sidl = BSTemplates.scandalTargets.Defaults('bs', self.filesets['sidl'])
    sidl.addServerLanguage('C++')
    sidl.addClientLanguage('C++')
    sidl.addClientLanguage('Python')
    compile = BSTemplates.compileTargets.Defaults(sidl)

    self.targets['sidl']    = sidl.getSIDLTarget()
    self.targets['compile'] = compile.getCompileTarget()
    self.targets['default'] = self.targets['compile']

if __name__ ==  '__main__':
#  try:
    pm = PetscMake(sys.argv[1:])
    pm.main()
    pm.install()
#  except Exception, e:
#    print 'ERROR: '+str(e)
#    sys.exit(1)
    sys.exit(0)
