#!/usr/bin/env python
import user
import build.buildGraph
import build.framework
import project

import os

class PetscMake(build.framework.Framework):
  def __init__(self, clArgs = None, argDB = None):
    build.framework.Framework.__init__(self, project.Project('bk://sidl.bkbits.net/BuildSystem', self.getRoot()), clArgs, argDB)
    self.project.setWebDirectory('petsc@harley.mcs.anl.gov://mcs/www-unix/ase')
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
    self.cpWebsite('docs/website/index.html')
    self.cpWebsite('install/bootstrap.py', 'bootstrap.sh')
    self.cpWebsite('docs/tutorials/*.ppt')
    self.cpWebsite('docs/website/faq.html')
    self.cpWebsite('docs/website/projects.html')
    return

  def t_updateBootstrap(self):
    import install.installerclass

    for url in ['bk://sidl.bkbits.net/Runtime', 'bk://sidl.bkbits.net/Compiler']:
      installer = install.installerclass.Installer()
      dir       = os.path.join('/mcs','ftp', 'pub', 'petsc', 'sidl')
      tarball   = installer.getRepositoryName(installer.getMappedUrl(url))+'.tgz'
      fullPath  = os.path.join(dir, tarball)
      installer.backup(url)
      try: self.executeShellCommand('ssh petsc@harley.mcs.anl.gov mv '+fullPath+' '+fullPath+'.old')
      except: pass
      self.cpFile(tarball, 'petsc@harley.mcs.anl.gov:/'+dir)
      os.remove(tarball)
    return
  
if __name__ ==  '__main__':
  import sys
  pm = PetscMake(sys.argv[1:])
  pm.main()
