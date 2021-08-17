#!/usr/bin/env python
from __future__ import absolute_import
import maker
import project

import os

class Make(maker.Make):
  def __init__(self):
    maker.Make.__init__(self)
    self.project = project.Project('https://bitbucket.org/petsc/buildsystem', self.getRoot())
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

if __name__ == '__main__':
  Make().run()
