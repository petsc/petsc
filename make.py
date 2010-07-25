#!/usr/bin/env python
import user
import maker
import project

import os

class Make(maker.Make):
  def __init__(self):
    maker.Make.__init__(self)
    self.project = project.Project('http://petsc.cs.iit.edu/petsc/BuildSystem', self.getRoot())
    self.project.setWebDirectory('petsc@login.mcs.anl.gov://mcs/www-unix/ase')
    return

  def setupDependencies(self, sourceDB):
    maker.Make.setupDependencies(self, sourceDB)
    sourceDB.addDependency(os.path.join('client-python', 'cygwinpath.c'), os.path.join('client-python', 'cygwinpath.h'))
    return

  def updateDependencies(self, sourceDB):
    sourceDB.updateSource(os.path.join('client-python', 'cygwinpath.h'))
    maker.Make.updateDependencies(self, sourceDB)
    return

  def setupConfigure(self, framework):
    doConfigure = maker.Make.setupConfigure(self, framework)
    framework.header = os.path.join('client-python', 'cygwinpath.h')
    return doConfigure

  def configure(self, builder):
    framework   = maker.Make.configure(self, builder)
    self.python = framework.require('config.python', None)
    return

  def buildCygwinPath(self, builder):
    '''Builds the Python module which translates Cygwin paths'''
    builder.pushConfiguration('Triangle Library')
    compiler = builder.getCompilerObject()
    linker   = builder.getLinkerObject()
    compiler.includeDirectories.update(self.python.include)
    linker.libraries.update(self.python.lib)
    source = os.path.join('client-python', 'cygwinpath.c')
    object = os.path.join('client-python', 'cygwinpath.o')
    self.builder.compile([source], object)
    self.builder.link([object], os.path.join('client-python', 'cygwinpath.so'), shared = 1)
    builder.popConfiguration()
    return

  def build(self, builder):
    self.buildCygwinPath(builder)
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
      try: self.executeShellCommand('ssh petsc@login.mcs.anl.gov mv '+fullPath+' '+fullPath+'.old')
      except: pass
      self.cpFile(tarball, 'petsc@login.mcs.anl.gov:/'+dir)
      os.remove(tarball)
    return

if __name__ == '__main__':
  Make().run()
