#!/usr/bin/env python
import user
import build.buildGraph
import build.framework
import project

import os

class PetscMake(build.framework.Framework):
  def __init__(self, clArgs = None, argDB = None):
    build.framework.Framework.__init__(self, project.Project('bs', 'bk://sidl.bkbits.net/BuildSystem', self.getRoot()), clArgs, argDB)
    return

  def setupSIDL(self):
    import build.fileset
    self.filesets['sidl'] = build.fileset.FileSet()
    return

  def setupSource(self):
    import build.fileset
    url                     = self.project.getUrl()
    pythonRoot              = self.sidlTemplate.usingSIDL.getClientRootDir('Python')
    self.filesets['python'] = build.fileset.RootedFileSet(url, [os.path.join(pythonRoot, 'cygwinpath_Module.c')], tag = 'python client c')
    self.filesets['sidl'].children.append(self.filesets['python'])
    return

  def setupProject(self):
    if not 'installedprojects'  in self.argDB:
      self.argDB['installedprojects']  = []
    if not 'installedLanguages' in self.argDB:
      self.argDB['installedLanguages'] = ['Python', 'Cxx']
    if not 'clientLanguages'    in self.argDB:
      self.argDB['clientLanguages']    = []
    self.setupSIDL()
    return

  def setupBuild(self):
    self.setupSource()
    self.sidlTemplate.addClient('Python')
    self.configureHeader = os.path.join(self.project.getRoot(), self.sidlTemplate.usingSIDL.getClientRootDir('Python', 'sidl'), 'cygwinpath_Module.h')
    return

if __name__ ==  '__main__':
  import sys
  pm = PetscMake(sys.argv[1:])
  pm.main()
