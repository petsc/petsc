#!/usr/bin/env python
from __future__ import generators
import user
import config.base
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.compilers    = self.framework.require('config.compilers',self)
    self.libraries    = self.framework.require('config.libraries',self)
    self.mpi          = self.framework.require('PETSc.packages.MPI',self)
    self.foundBS95    = 0
    self.lib          = ''
    self.include      = ''
    return

  def __str__(self):
    if self.foundBS95: return 'BlockSolve95: Using '+str(self.lib)+'\n'
    return ''
  
  def configureHelp(self,help):
    import nargs
    help.addArgument('BLOCKSOLVE95','-with-blocksolve95=<lib>',nargs.Arg(None,None,'Indicate the library containing BlockSolve95'))
    help.addArgument('BLOCKSOLVE95','-with-blocksolve95-dir=<dir>',nargs.Arg(None,None,'Indicate the root of the BlockSolve95 installation'))
    help.addArgument('BLOCKSOLVE95','-with-blocksolve95-bopt=<bopt>',nargs.Arg(None,None,'Indicate the BlockSolve95 bopt to use'))
    help.addArgument('BLOCKSOLVE95','-with-blocksolve95-arch=<arch>',nargs.Arg(None,None,'Indicate the BlockSolve95 arch to use'))
    return

  def generateGuesses(self):
    if 'with-blocksolve95' in self.framework.argDB:
      yield ('User specified BLOCKSOLVE95 library',self.framework.argDB['with-blocksolve95'])
      if 'with-blocksolve95-dir' in self.framework.argDB and 'with-blocksolve95-bopt' in self.framework.argDB and 'with-blocksolve95-arch' in self.framework.argDB:
        dir    = self.framework.argDB['with-blocksolve95-dir']
        bopt   = self.framework.argDB['with-blocksolve95-bopt']
        bsarch = self.framework.argDB['with-blocksolve95-arch']
        yield('User specified BLOCKSOLVE95 installation',os.path.join(dir,'lib','lib'+bopt,bsarch,'libBS95.a'))
        yield('User specified BLOCKSOLVE95 installation',os.path.join(dir,'lib','lib'+bopt,bsarch,'libBS95.lib'))
      # Perhaps we could also check all possible blocksolve95 installations based on just with-blocksolve95-dir trying all bopt and bsarch available....

  def checkLib(self,bs95lib):
    if not isinstance(bs95lib,list): bs95lib = [bs95lib]
    oldLibs = self.framework.argDB['LIBS']
    # This next line is really ugly
    # ' '.join(map(self.libraries.getLibArgument',self.mpi.lib)
    # takes the location of the MPI library and separates it into a string that looks like:
    # -L<MPI_DIR> -l<mpi library>
    self.foundBS95 = self.libraries.check(bs95lib,'BSinit',otherLibs=' '.join(map(self.libraries.getLibArgument, self.mpi.lib)))
    if self.foundBS95:
      self.lib    = bs95lib
      self.framework.log.write('Found functional BlockSolve95: '+str(self.lib)+'\n')
      # This next stuff to generate the include should actually be testing existance of the actual headers and also check with-blocksolve95-include
      # This is just quick and dirty -- really 
      bsroot = self.lib[0]
      # We have /home/user/BlockSolve95/lib/libO/bsarch/libBS95.a so remove the last 4 elements from the path
      for i in 1,2,3,4:
        (bsroot,dummy) = os.path.split(bsroot)
      self.include = os.path.join(bsroot,'include')
      self.addSubstitution('BLOCKSOLVE_INCLUDE','-I'+self.include)
      self.addSubstitution('BLOCKSOLVE_LIB',' '.join(map(self.libraries.getLibArgument,self.lib)))
      self.addDefine('HAVE_BLOCKSOLVE',1)
    self.framework.argDB['LIBS']=oldLibs
    return(self.foundBS95)

  def configureLibrary(self):
    '''Find a Blocksolve installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    for (configstr,bs95lib) in self.generateGuesses():
      self.framework.log.write('Checking for a functional BlockSolve95 in '+configstr+'\n')
      if self.executeTest(self.checkLib,bs95lib):
        self.foundBS95 = 1
        break
    return

  def setEmptyOutput(self):
    #self.addDefine('HAVE_BLOCKSOLVE', 0)
    self.addSubstitution('BLOCKSOLVE_INCLUDE', '')
    self.addSubstitution('BLOCKSOLVE_LIB', '')
    return

  def configure(self):
    if not 'with-blocksolve95' in self.framework.argDB:
      self.setEmptyOutput()
      return
    self.executeTest(self.configureLibrary)
    return

if __name__ == '__main__':
  import config.framework
  import sys
  framework = config.framework.Framework(sys.argv[1:])
  framework.setupLogging()
  framework.children.append(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
