#!/usr/bin/env python
import user
import build.buildGraph
import build.framework
import project

import os

class PetscMake(build.framework.Framework):
  def __init__(self, clArgs = None, argDB = None):
    build.framework.Framework.__init__(self, project.Project('bk://sidl.bkbits.net/BuildSystem', self.getRoot()), clArgs, argDB)
    self.project.setWebDirectory('petsc@terra.mcs.anl.gov://mcs/www-unix/sidl')
    return

  def t_configure(self):
    self.configureHeader = os.path.join(self.project.getRoot(), self.sidlTemplate.usingSIDL.getClientRootDir('Python', 'sidl'), 'cygwinpath_Module.h')
    return build.framework.Framework.t_configure(self)

  def setupSIDL(self):
    import build.fileset
    self.filesets['sidl'] = build.fileset.FileSet()
    return

  def setupSource(self):
    import build.fileset
    url                     = self.project.getUrl()
    pythonRoot              = self.sidlTemplate.usingSIDL.getClientRootDir('Python')
    self.filesets['python'] = build.fileset.RootedFileSet(url, [os.path.join(pythonRoot, 'cygwinpath_Module.c')], tag = 'python client')
    self.filesets['sidl'].children.append(self.filesets['python'])
    return

  def setupProject(self):
    self.setupSIDL()
    return

  def setupBuild(self):
    self.setupSource()
    self.sidlTemplate.addClient('Python')
    return

  def t_updateWebsite(self):
    build.framework.Framework.t_updateWebsite(self)
    self.executeShellCommand('scp docs/website/index.html '+self.project.getWebDirectory()+'/index.html')
    self.executeShellCommand('scp install/bootstrap.py '+self.project.getWebDirectory()+'/bootstrap.sh')
    self.executeShellCommand('scp docs/tutorials/*.ppt '+self.project.getWebDirectory())
       
if __name__ ==  '__main__':
  import sys
  pm = PetscMake(sys.argv[1:])
  pm.main()
