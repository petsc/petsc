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
    output=''
    if self.foundBS95:
      output  = 'BlockSolve95:\n'
      output += '  Includes: '+ str(self.include)+'\n'
      output += '  Library: '+str(self.lib)+'\n'
    return output
  
  def configureHelp(self,help):
    import nargs
    help.addArgument('BLOCKSOLVE95','-with-blocksolve95=<bool>',nargs.ArgBool(None,1,'Indicate if you wish to test for BlockSolve95'))
    help.addArgument('BLOCKSOLVE95','-with-blocksolve95-lib=<lib>',nargs.Arg(None,None,'Indicate the library containing BlockSolve95'))
    help.addArgument('BLOCKSOLVE95','-with-blocksolve95-include=<lib>',nargs.ArgDir(None,None,'Indicate the directory for BlockSolve95 header files'))
    help.addArgument('BLOCKSOLVE95','-with-blocksolve95-dir=<dir>',nargs.ArgDir(None,None,'Indicate the root of the BlockSolve95 installation'))
    help.addArgument('BLOCKSOLVE95','-with-blocksolve95-bopt=<bopt>',nargs.Arg(None,None,'Indicate the BlockSolve95 bopt to use'))
    help.addArgument('BLOCKSOLVE95','-with-blocksolve95-arch=<arch>',nargs.Arg(None,None,'Indicate the BlockSolve95 arch to use'))
    return

  def generateIncludeGuesses(self):
    if 'with-blocksolve95-include' in self.framework.argDB:
      yield('User specified BLOCKSOLVE95 header location',self.framework.argDB['with-blocksolve95-include'])
    bsroot = self.lib[0]
    # We have /home/user/BlockSolve95/lib/libO/bsarch/libBS95.a so remove the last 4 elements from the path
    for i in 1,2,3,4:
      (bsroot,dummy) = os.path.split(bsroot)
    yield('based on found library location',os.path.join(bsroot,'include'))
    return

  def checkInclude(self,bs95incl):
    '''Check that BSsparse.h is present'''
    if not isinstance(bs95incl,list):bs95incl = [bs95incl]
    oldFlags = self.framework.argDB['CPPFLAGS']
    for inc in bs95incl:
      if not self.mpi.include is None:
        mpiincl = ' -I' + ' -I'.join(self.mpi.include)
      self.framework.argDB['CPPFLAGS'] += ' -I'+inc+mpiincl
    found = self.checkPreprocess('#include <BSsparse.h>\n')
    self.framework.argDB['CPPFLAGS'] = oldFlags
    if found:
      self.include = bs95incl
      self.framework.log.write('Found BlockSolve95 header file BSsparse.h: '+str(self.include)+'\n')
    return found

  def generateLibGuesses(self):
    if 'with-blocksolve95-lib' in self.framework.argDB:
      yield ('User specified BLOCKSOLVE95 library',self.framework.argDB['with-blocksolve95-lib'])
    elif 'with-blocksolve95-dir' in self.framework.argDB:
      if not 'with-blocksolve95-bopt' in self.framework.argDB:
        self.framework.log.write('Missing BOPT for specified BlockSolve root directory\n')
      elif not 'with-blocksolve95-arch' in self.framework.argDB:
        self.framework.log.write('Missing ARCH for specified BlockSolve root directory\n')
      else:
        dir    = self.framework.argDB['with-blocksolve95-dir']
        bopt   = self.framework.argDB['with-blocksolve95-bopt']
        bsarch = self.framework.argDB['with-blocksolve95-arch']
        yield('User specified BLOCKSOLVE95 installation',os.path.join(dir,'lib','lib'+bopt,bsarch,'libBS95.a'))
        yield('User specified BLOCKSOLVE95 installation',os.path.join(dir,'lib','lib'+bopt,bsarch,'libBS95.lib'))
    else:
      self.framework.log.write('Must specify either a library or installation root directory for BlockSolve\n')
      # Perhaps we could also check all possible blocksolve95 installations based on just with-blocksolve95-dir trying all bopt and bsarch available....
    return

  def checkLib(self,bs95lib):
    if not isinstance(bs95lib,list): bs95lib = [bs95lib]
    oldLibs = self.framework.argDB['LIBS']
    # This next line is really ugly
    # ' '.join(map(self.libraries.getLibArgument',self.mpi.lib)
    # takes the location of the MPI library and separates it into a string that looks like:
    # -L<MPI_DIR> -l<mpi library>
    found = self.libraries.check(bs95lib,'BSinit',otherLibs=' '.join(map(self.libraries.getLibArgument, self.mpi.lib)))
    self.framework.argDB['LIBS']=oldLibs
    if found:
      self.lib    = bs95lib
      self.framework.log.write('Found functional BlockSolve95: '+str(self.lib)+'\n')
    return found

  def configureLibrary(self):
    '''Find a Blocksolve installation and check if it can work with PETSc'''
    self.framework.log.write('==================================================================================\n')
    found = 0
    for (configstr,bs95lib) in self.generateLibGuesses():
      self.framework.log.write('Checking for a functional BlockSolve95 in '+configstr+'\n')
      found = self.executeTest(self.checkLib,bs95lib)
      if found: break
    if found:
      for (inclstr,bs95incl) in self.generateIncludeGuesses():
        self.framework.log.write('Checking for BlockSolve95 headers in '+inclstr+': '+bs95incl + '\n')
        if self.executeTest(self.checkInclude,bs95incl):
          self.include = bs95incl
          self.foundBS95 = 1
          self.setFoundOutput()
          break
    else:
      self.framework.log.write('Could not find a functional BlockSolve95\n')
      self.setEmptyOutput()
    return

  def setFoundOutput(self):
    self.addSubstitution('BLOCKSOLVE_INCLUDE','-I'+self.include)
    self.addSubstitution('BLOCKSOLVE_LIB',' '.join(map(self.libraries.getLibArgument,self.lib)))
    self.addDefine('HAVE_BLOCKSOLVE',1)
    
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
